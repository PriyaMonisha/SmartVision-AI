# SmartVision AI — Interview Preparation Guide
# Questions from India's Big Tech Companies

**Target companies:** HCL, TCS, Wipro, Infosys, Cognizant, Capgemini | Flipkart, Amazon India, Zomato, Swiggy | Google India, Microsoft India

---

## How to Use This File

- **Service companies (HCL/TCS/Wipro):** Focus on Sections 1–5 + 9. They ask concepts + project walkthrough.
- **Product companies (Flipkart/Amazon/Zomato):** Focus on Sections 3–6 + 8. They go deep on deep learning fundamentals and system design.
- **MNC product (Google/Microsoft):** Focus on Sections 4–8 + 12. They push on theory, scalability, and trade-offs.
- For every answer, use the **STAR format** (Situation → Task → Action → Result) when describing your work.

---

## Section 1 — Project Introduction (HR + First Technical Round)

**Q1. Tell me about your SmartVision AI project in 2 minutes.**

> SmartVision AI is a production-grade multi-class object recognition platform I built as a GUVI HCL capstone project. It does two things: classify objects into 25 COCO categories using 4 pretrained CNN models — VGG16, ResNet50, MobileNetV2, and EfficientNetB0 — and detect multiple objects in a scene using YOLOv8. All 4 CNNs are trained using transfer learning and fine-tuning on a curated 2,500-image dataset streamed directly from HuggingFace. The production stack is: FastAPI serves all model inference, Redis caches results, Prometheus scrapes metrics, Grafana dashboards them, and a Streamlit app provides the user interface — calling FastAPI endpoints only, never loading models directly. Training happened on Google Colab T4, weights are stored on HuggingFace Hub, and the full serving stack runs in Docker Compose. MLflow tracks every training run.

**Q2. Why object recognition? Why the COCO dataset?**

> COCO (Common Objects in Context) is the industry benchmark for object detection and classification — it covers 80 real-world classes with high-quality annotations. I chose a 25-class subset covering the most practically useful categories: vehicles, animals, furniture, food, and people. COCO is also freely accessible via HuggingFace in streaming mode, which eliminates the need to download 165GB and makes the data pipeline reproducible on any machine. The goal was to demonstrate a full production CV pipeline, not to invent a novel dataset.

**Q3. What is the business value? Who would use this?**

> Any application needing real-time object identification: retail inventory (recognise products on shelves), smart parking (count cars, trucks, buses), accessibility tools (describe objects to visually impaired users), or security cameras (identify people, animals, vehicles). The architecture is designed for production: responses are cached, inference is sub-100ms on GPU, drift is monitored so you know when to retrain, and the full stack deploys with one `docker compose up`.

---

## Section 2 — Dataset & Data Pipeline Questions

**Q4. COCO has 165GB. How did you get the data without downloading everything?**

> I used HuggingFace's streaming API: `load_dataset("detection-datasets/coco", streaming=True)`. In streaming mode, the dataset is an iterator — it fetches images on demand, one batch at a time, without downloading the full dataset to disk. I streamed through ~10,253 images and extracted exactly 100 per class (2,500 total) with a safety limit of MAX_ITER=60,000 in case some classes are rare. The trade-off is slower per-image throughput due to network latency, but for a one-time data preparation step that's acceptable. The entire collection took about 15–20 minutes on Colab.

**Q5. What happens if Colab disconnects mid-download? You'd lose everything.**

> I implemented checkpoint/resume logic. After each class reaches its target count, `save_checkpoint(progress)` writes a JSON file (`download_progress.json`) mapping class name → count collected. On re-run, `load_checkpoint()` reads that file, identifies which classes are already complete, and skips them. The streaming loop only fetches images for remaining classes. If Colab disconnects after collecting 18 of 25 classes, the next run collects only the 7 missing ones — no wasted work.

**Q6. Your mentor's notebook had 26 classes. Why do you have 25?**

> The mentor's notebook accidentally included the 'train' vehicle class (HuggingFace category ID 6) alongside the 25 specified in the project PDF. This caused `nc: 26` in their YOLO data.yaml — a silent bug that would produce wrong class counts in detection training. I identified this by reading the project specification carefully against the mentor's `SELECTED_CLASSES` dict. Our `verify_coco_mapping()` function (called as the very first step in data acquisition) asserts `len(SELECTED_CLASSES) == NUM_CLASSES == 25` and fails immediately if there's a mismatch — catching this class before a single image is downloaded.

**Q7. What is the COCO ID problem? Why can't you use COCO category IDs directly?**

> The original COCO annotations use non-sequential IDs from 1 to 90 with gaps (there is no category 12, 26, 29, etc. — those were never defined). HuggingFace's `detection-datasets/coco` re-indexes categories to 0–79 contiguously. So a `car` in the original COCO has ID 3, but in HuggingFace it has ID 2. If you hardcode original COCO IDs and stream from HuggingFace, every category lookup fails silently — you'd collect zero images for car and never know why. Our `HF_CATEGORY_TO_CLASS_IDX` mapping in `config.py` uses the correct HuggingFace 0-indexed IDs verified against the actual dataset schema.

**Q8. How does `verify_coco_mapping()` work? Why is it called first?**

> It performs a round-trip consistency check before streaming a single image:
> 1. Asserts `len(SELECTED_CLASSES) == 25` (catches mentor's 26-class bug)
> 2. Asserts `len(HF_CATEGORY_TO_CLASS_IDX) == 25` (no duplicate or missing IDs)
> 3. For each class: `class_name → hf_id → class_idx → class_name` must round-trip perfectly
> 4. Asserts no duplicate HF IDs exist across classes
>
> It's called first because streaming 60,000 images with a wrong mapping produces zero useful data — you'd waste 20 minutes before discovering the bug. Fail fast, fail loud.

**Q9. Why are classification images cropped 224×224 objects, but detection images are full scenes?**

> Two different tasks, two different model inputs:
>
> **Classification** asks "what is this object?" — it needs an isolated crop containing just one object. The crop comes directly from the COCO bounding box: `img.crop((x, y, x+w, y+h))` then `resize(224, 224)`. This gives the CNN a clean, object-centred input matching ImageNet's training format.
>
> **Detection** asks "where are all the objects in this scene?" — it needs the full image with spatial context. YOLOv8 processes 640×640 full images and predicts bounding boxes for multiple objects simultaneously. Cropping first would destroy the spatial relationships the detector needs to learn.

**Q10. How do you convert COCO bounding boxes to YOLO format?**

> COCO stores bboxes as `[x, y, w, h]` in absolute pixels (top-left corner + width/height). YOLO requires normalized centre coordinates: `[class_idx, x_center, y_center, w_norm, h_norm]` where all values are in [0, 1] relative to image dimensions.

```python
def bbox_to_yolo(bbox, img_w, img_h, class_idx):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w    # Centre x, normalized
    y_center = (y + h / 2) / img_h    # Centre y, normalized
    w_norm   = w / img_w
    h_norm   = h / img_h

    # Clamp to valid range; w_norm/h_norm minimum 0.001 (no zero-size boxes)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm   = max(0.001, min(1.0, w_norm))
    h_norm   = max(0.001, min(1.0, h_norm))

    return f"{int(class_idx)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
```

> The clamping prevents negative coordinates (which happen when COCO annotations slightly exceed image boundaries) and zero-size boxes (which crash YOLO training).

---

## Section 3 — CNN Architecture & Transfer Learning Questions

**Q11. You trained 4 models. Compare them — architecture, size, strategy.**

| Model | Pretrained Weights | Head Input | Head Output | Model Size | Batch | Epochs | Strategy |
|---|---|---|---|---|---|---|---|
| VGG16 | ImageNet1K_V1 | 4096 | 25 | ~550MB | 16 | 20 (5+15) | 2-phase fine-tune |
| ResNet50 | ImageNet1K_V2 | 2048 | 25 | ~100MB | 32 | 25 (6+19) | 2-phase fine-tune |
| MobileNetV2 | ImageNet1K_V1 | 1280 | 25 | ~14MB | 64 | 20 | Single-phase |
| EfficientNetB0 | ImageNet1K_V1 | 1280 | 25 | ~20MB | 32 | 25 | Single-phase + AMP |

> All 4 models: input `(B, 3, 224, 224)`, ImageNet-normalized. Head replacement replaces only the final linear layer — everything else is pretrained.

**Q12. What is transfer learning? Why use pretrained ImageNet weights?**

> Transfer learning reuses a model trained on a large dataset (ImageNet: 1.28M images, 1000 classes) as a starting point for a new task. The intuition: early convolutional layers learn general visual features (edges, textures, colours) that are useful for any image recognition task. Only the later layers need to specialise for the new classes. Training from scratch on 2,500 images would overfit badly — ImageNet pretrained weights give us a vastly better starting point. The 25-class head (the final linear layer) is randomly initialised and trained from scratch on our data.

**Q13. Why do VGG16 and ResNet50 use 2-phase training, but MobileNetV2 and EfficientNetB0 don't?**

> Two factors: architecture size and dataset size.
>
> **VGG16 has 134M parameters; ResNet50 has 25M**. With only 1,750 training images, fine-tuning all parameters at once causes catastrophic forgetting — the pretrained features get overwritten by noise from the small dataset. 2-phase training prevents this: Phase 1 warms up the new head safely with all conv weights frozen, then Phase 2 carefully unlocks just the last conv block at a lower LR.
>
> **MobileNetV2 has 3.5M parameters; EfficientNetB0 has 5.3M**. These lightweight architectures are designed for transfer learning on small datasets — their features generalise better. Head-only training gives 80%+ accuracy without fine-tuning, so the added complexity of 2-phase isn't needed.

**Q14. Walk me through VGG16's 2-phase training in detail.**

> **Phase 1 — Head only (5 epochs):**
> - All 134M parameters frozen except `classifier[6]` (the new 4096→25 head = 102,425 params)
> - Optimizer: Adam, lr=0.001, weight_decay=1e-4
> - Scheduler: StepLR(step_size=3, gamma=0.1) — LR decays every 3 epochs
> - Early stopping: patience=5 epochs
> - Purpose: Let the randomly initialised head converge to reasonable weights without corrupting ImageNet conv features
>
> **Phase 2 — Fine-tune last conv block (15 epochs):**
> - Unfreeze `features[24:]` — 7 layers (3× Conv2d(512,512), 3× ReLU, 1× MaxPool2d) ≈ 7.1M extra params
> - Total trainable: ~102K + ~7.1M ≈ 7.2M (5.4% of 134M)
> - Optimizer: Adam, lr=0.0001 (10× lower than Phase 1), weight_decay=1e-4
> - Scheduler: CosineAnnealingLR(T_max=15) — smoother decay for fine-tuning
> - Early stopping: patience=8 epochs (longer — Phase 2 needs more time to converge)
> - Purpose: Adapt low-level conv features (edges, textures) from ImageNet classes to 25 COCO classes

**Q15. VGG16 got 59.2% with head-only training. That's below your 80% threshold. What did you do?**

> I diagnosed the root cause before changing anything. The output showed `Trainable params: 102,425 / 134,362,969 (0.1%)`. With only 0.1% of parameters training, and VGG16's frozen features optimised for 1000 ImageNet classes (not 25 COCO classes), the head couldn't compensate. The features being passed to the classifier didn't represent our classes well enough.
>
> The fix: implement 2-phase training — unfreeze `features[24:]` (last conv block, ~7.1M params) in Phase 2 at lr=0.0001. Phase 2 ran 15 epochs with CosineAnnealingLR. Phase 2 trainable params confirmed: 7,181,849 (unfreezing verified before training started).
>
> **Phase 2 result**: train accuracy climbed to 92%. Val accuracy plateaued at 59.5% — a 33% train-val gap. This is textbook overfitting. (See Q15a for what that means and the decision made.)

**Q15a. You implemented 2-phase training. Phase 2 train accuracy reached 92% but val accuracy stayed at 59.5%. What does that pattern tell you? What did you do?**

> **What it means**: A 33% train-val gap is textbook overfitting — the model memorised the 1,725 training images but didn't learn generalisable features. Val accuracy plateauing at 59.5% (same as Phase 1) means Phase 2 fine-tuning added capacity without adding generalisation.
>
> **Root cause — params/image ratio**: VGG16's last conv block has ~7.1M parameters. With 1,725 training images: `7,100,000 / 1,725 ≈ 4,100 params per image`. Senior ML rule of thumb: when params-per-image exceeds ~1,000, the model has enough capacity to memorise the training set rather than learn from it. For comparison, MobileNetV2's last block has 300K params (174 params/image) — 24× less overfitting risk.
>
> **The decision**: Accept VGG16 at 59.5% and document it as an architecture limitation. This is a deliberate engineering call, not a failure:
> - VGG16 (2014) is architecturally obsolete for transfer learning on small datasets — no skip connections, too many params in late layers
> - Spending 3+ more hours tuning VGG16 to reach 72-75% is a poor ROI vs spending 1 hour training ResNet50 to 82-88%
> - The codebase already has correct 2-phase patterns, and this validated that the training pipeline works
>
> **Written in the experiment log**: "VGG16: 59.5% val accuracy. Overfitting in Phase 2 (train 92%, val 59%, 33% gap). Architecture limitation: 7.1M last-block params on 1,725 images = 4,100 params/image. Moving to ResNet50 per industry best practice."

**Q16. What is catastrophic forgetting? How does your training strategy prevent it?**

> Catastrophic forgetting is when fine-tuning a pretrained network on a new task causes it to lose its previously learned representations — the new gradient updates overwrite the old weights. It's worst when you fine-tune everything at once with a high learning rate.
>
> Our prevention strategy: (1) Start with features frozen (Phase 1) — no gradient flows into conv layers, so pretrained features are preserved. (2) In Phase 2, use lr=0.0001 instead of 0.001 — small updates that shift features gradually rather than overwriting them. (3) CosineAnnealingLR in Phase 2 — starts moderate and decays smoothly, preventing abrupt large updates late in training.

**Q17. Why is VGG16's batch size 16 when other models use 32 or 64?**

> Memory. VGG16 weights alone are ~550MB on GPU. A single forward pass also needs activation memory for backpropagation. At batch=32, the peak GPU memory during VGG16 training exceeds the T4's 16GB — you get CUDA out-of-memory errors. At batch=16, peak memory stays within budget. MobileNetV2 (14MB) and EfficientNetB0 (20MB) can comfortably use batch=64 and batch=32 respectively — their weights take 30–40× less memory.

**Q17a. ResNet50 Phase 2 (layer3+layer4 unfrozen) also overfit — train 99%, val 70%, 29% gap. Same diagnosis as VGG16. What did you change?**

> The pattern was identical to VGG16 but at a larger scale. Calculated params/image: `24,000,000 / 1,725 = 13,913` — 3× worse than VGG16's 4,174. The root cause was unfreezing too many layers at once.
>
> Fix: changed `unfreeze_resnet50_phase2()` to unfreeze **only layer4** (the last residual block), keeping layer3 frozen. This drops trainable params from ~24M to ~16M (9,275 params/image). The docstring now explicitly records the reason: "layer3+layer4 = 24M params caused overfitting on 1,725 images."
>
> **Why layer4 and not layer3**: layer4 is the highest-level feature extractor — closest to the classification head, most relevant to the output. Layer3 learns mid-level features that transfer well from ImageNet. Unfreezing it adds params without adding value for a 25-class task.
>
> **The diagnostic framework I applied across both models:**
> 1. Calculate params/image ratio (simple division)
> 2. If > ~5,000 → overfitting risk is high → reduce unfrozen layers
> 3. If train-val gap > 20% after Phase 2 → confirm overfitting → act
>
> This is a repeatable, quantitative rule — not guesswork.

---

## Section 4 — YOLOv8 Object Detection Questions

**Q18. What is the difference between classification and object detection architecturally?**

> **Classification** takes one image, outputs one class label (and confidence scores for all classes). The CNN processes the whole image as a fixed-size input and produces a single prediction vector of length N_classes.
>
> **Object detection** takes one image, outputs variable-length predictions: a list of `(class_id, x_center, y_center, width, height, confidence)` for each detected object. YOLOv8 uses a backbone (feature extraction), a neck (multi-scale feature aggregation with FPN), and a head (predicts boxes + classes at multiple scales). It can detect multiple objects of multiple classes in a single forward pass.

**Q19. Why does YOLO use normalized coordinates (0–1) instead of pixel coordinates?**

> Two reasons: (1) **Resolution independence** — if you resize the image from 640×640 to 320×320, normalized coordinates stay the same (0.5, 0.3 is still the centre regardless of resolution). Pixel coordinates would need to be rescaled. (2) **Training stability** — pixel coordinates for a 640×640 image range from 0 to 640. Normalized values (0–1) keep all coordinate targets in the same range, which is more numerically stable for gradient descent.

**Q20. Why did you use train+val images for detection, not test images?**

> Test images are held out for unbiased evaluation. Using them in detection training would cause data leakage — the model would have seen the test data during training, and test metrics would be optimistically biased. The detection dataset uses only train and val images from the classification pipeline, maintaining a clean test set for final evaluation.

**Q21. What is IOU? What is NMS? Why does YOLO need both?**

> **IOU (Intersection over Union)**: measures overlap between two bounding boxes. `IOU = Area_of_Intersection / Area_of_Union`. Range [0, 1]: 0 = no overlap, 1 = perfect overlap. Used to evaluate detection quality and as a threshold in NMS.
>
> **NMS (Non-Maximum Suppression)**: YOLO generates hundreds of candidate boxes for each object (one per grid cell). NMS keeps only the best-scoring box for each object and suppresses redundant overlapping boxes. Algorithm: sort boxes by confidence, keep the highest-confidence box, discard all boxes with IOU > threshold (0.45) against it, repeat.
>
> **Why both**: IOU measures overlap, NMS uses IOU to remove duplicates. Without NMS, every detected car might have 20 overlapping boxes around it. Our IOU threshold of 0.45 means boxes overlapping more than 45% are considered duplicates.

**Q22. What does `exist_ok=True` do in YOLO training?**

> `model.train(exist_ok=True)` tells YOLOv8 not to crash if the output directory already exists. Without it, YOLO raises an error if you re-run training (the directory from the previous run is still there). With `exist_ok=True`, it overwrites cleanly. This is Rule 8 in our project — mandatory to avoid brittle training runs.

**Q22a. Your YOLOv8n achieved mAP50=14.7%. Is that good or bad? What's the root cause of the ceiling?**

> In absolute terms, 14.7% is modest — but it's the expected result given the data size, and it's the honest answer.
>
> **Context**: YOLOv8n on the full COCO dataset (118,287 training images, 80 classes) achieves 52.3% mAP50. We trained on 3,080 images across 22 classes (~140 images/class). That's 9.5% of COCO's training volume per class. The mAP50 we got (14.7%) is proportional to the data deficit.
>
> **Root cause**: Detection is harder than classification — the model must learn both "what is this?" AND "where is it and how large is it?". This requires more training examples per class than classification. 140 img/class is enough for classification (we got 65.5% with ResNet50) but insufficient for detection.
>
> **Per-class evidence**: Classes with clean, visually distinctive objects scored highest — cat (50.1%), pizza (34.0%), bed (32.5%), airplane (34.0%). Classes with cluttered scenes or ambiguous sizes scored near zero — bicycle (4.8%), bottle (3.9%), traffic light (0%). This pattern is consistent with data-limited detection: the model learned only the most distinctive objects.
>
> **What would improve it**: 300–400 images/class would likely push mAP50 to 30–40%. The architecture (YOLOv8n, 3M params) is not the bottleneck — data volume is.

**Q22b. Why did you validate annotations before training? What did you find?**

> Annotation validation runs before training to fail fast on corrupt data rather than waste 90 minutes of GPU time before discovering errors in epoch 50.
>
> Our validator checks every `.txt` label file for: (1) class ID in valid range [0, 21], (2) bounding box coordinates in [0, 1], (3) non-zero box dimensions. We found 10,432 "errors" — all of the form `xc=1.0` or `yc=1.0` (centre coordinates at the image edge).
>
> **Root cause**: COCO has objects that extend to or past the image boundary. When converting from COCO format (`x, y, w, h` absolute pixels) to YOLO format (normalized centre coordinates), clamping clips these to exactly 1.0. This is valid YOLO format — YOLO clips coordinates during training anyway. Our initial validator used strict `< 1.0`; the fix was `<= 1.0` (closed interval). The annotations were correct all along.
>
> **Why this matters in interviews**: I validated BEFORE training, found a false positive in my own validator, diagnosed the root cause (COCO border objects), and fixed the validator logic — not the data. This is the correct debugging approach.

**Q22c. Why YOLOv8n (nano) and not YOLOv8s/m/l?**

> Three reasons: (1) **Data size** — with 140 img/class, larger models would overfit worse than the nano variant. More capacity means more overfitting on sparse data. (2) **Training time** — YOLOv8n (3M params) trains 50 epochs in 97 minutes on T4. YOLOv8m (25M params) would take ~4× longer with no benefit given our data ceiling. (3) **Deployment** — the full serving stack loads 4 CNN classifiers + YOLOv8 simultaneously. YOLOv8n is only 6MB vs 49MB for YOLOv8m, fitting easily within our ~1.5GB Docker memory budget.

---

## Section 5 — Model Training & Optimization Questions

**Q23. What is early stopping? Why is Phase 1 patience=5 but Phase 2 patience=8?**

> Early stopping monitors validation accuracy each epoch and stops training if it hasn't improved for `patience` epochs. This prevents overfitting — if val accuracy plateaus, continuing training just memorizes training data.
>
> **Phase 1 patience=5**: Head-only training converges fast (only 102K parameters). If it doesn't improve for 5 epochs, it's converged. No reason to wait longer.
>
> **Phase 2 patience=8**: Fine-tuning larger parameter sets has more complex loss landscapes and slower convergence. The optimizer may explore a flat region for a few epochs before finding a better direction. Patience=8 gives it room to escape local plateaus without stopping too early.

**Q24. You used StepLR in Phase 1 and CosineAnnealingLR in Phase 2. Why different schedulers?**

> **StepLR(step_size=3, gamma=0.1)**: Drops LR by 10× every 3 epochs. Good for Phase 1 (head training) because the head needs aggressive early learning, then sharp drops to stabilise convergence.
>
> **CosineAnnealingLR(T_max=15)**: Smoothly anneals LR from initial value to near-zero following a cosine curve. Good for fine-tuning because: (1) gradual decay avoids overshoot, (2) the smooth curve allows the optimizer to escape local minima near the end of training, (3) it's been empirically shown to outperform step decay for fine-tuning tasks.

**Q25. EfficientNetB0 uses mixed precision. Explain what that means and why only EfficientNet?**

> **Mixed precision**: combines float16 (half precision) for forward pass computations with float32 (full precision) for loss and gradients.
>
> ```python
> scaler = torch.cuda.amp.GradScaler()  # Prevents gradient underflow
> with torch.autocast("cuda"):           # Forward in float16
>     output = model(images)
>     loss = criterion(output, labels)
> scaler.scale(loss).backward()          # Scaled backward in float32
> scaler.step(optimizer)
> scaler.update()
> ```
>
> **Benefits**: ~50% less GPU memory for activations, ~2× faster on tensor cores.
>
> **Why EfficientNet only**: EfficientNetB0 is architecturally designed for efficient computation (compound scaling). Mixed precision with EfficientNet is stable and well-tested. VGG16's sequential, dense architecture doesn't benefit as much, and adding AMP complexity to Phase 2 fine-tuning introduces instability risk we didn't need.

**Q26. Why Adam optimizer across all 4 models? Why not SGD?**

> Adam is adaptive — it computes per-parameter learning rates using first and second moment estimates of gradients. For transfer learning, this is crucial: different layers need different effective learning rates (frozen-then-unfrozen layers, new head vs pretrained features). SGD uses a single global LR and requires careful tuning of momentum and LR schedule to work well. Adam converges faster with less hyperparameter sensitivity, which matters when training 4 models on limited Colab time. The trade-off: Adam may generalise slightly worse than well-tuned SGD on very large datasets, but on 2,500 images the difference is negligible.

**Q26b. Why did MobileNetV2 Round 2 produce 62.3% test accuracy (well below the 76-83% target)?**

> Three contributing factors, ordered by impact:
>
> **1. Backbone learning rate too high (highest impact):** Round 2 used lr=1e-4 for ALL unfrozen parameters — backbone and head alike. At lr=1e-4, pretrained ImageNet features update fast enough to overfit the training set before the head can generalize. The fix: **differential learning rate** — backbone lr=1e-5 (100x lower), head lr=1e-4. Pretrained features need gentle nudges, not large steps.
>
> **2. Too many unfrozen parameters (medium impact):** features[14:] exposed 1.71M / 3,080 training images = 552 parameters per training image. Rule of thumb: > 500 params/image = high overfit risk. Changed to features[16:]: 1.23M / 3,080 = 401 params/image — same fine-tuning signal with 28% less memorization capacity.
>
> **3. Insufficient data for this problem complexity (data ceiling):** 22-class fine-grained COCO crop classification with household items (cup/bowl/bottle all look similar) requires at least 300 samples/class for MobileNetV2 to generalize well. With 140 training samples/class, the realistic ceiling is 68-74%, not 76-83%.
>
> **Evidence**: 27.4pp train/val gap (88.5% train vs 61.1% val) = textbook memorization. Phase 2 added only 4.4pp val accuracy over Phase 1, while adding 1.71M parameters — near-zero return on added capacity.

**Q26c. What is AdamW and why is it better than Adam for fine-tuning pretrained models?**

> **Adam's weight decay problem**: Standard Adam applies weight decay INSIDE the adaptive moment update. The effective decay for each parameter is: `wd / sqrt(v_hat + eps)` — it's scaled by the inverse of the gradient magnitude. Fast-updating parameters get weaker effective decay; slow-updating parameters get stronger. This is inconsistent and mathematically not the intended L2 regularization.
>
> **AdamW fix**: Decouple weight decay from the gradient update. The update is:
> ```
> theta_t = theta - lr * m_hat/sqrt(v_hat) - lr * wd * theta
> ```
> Weight decay is applied directly to the parameter, independent of gradient history.
>
> **Why it matters for fine-tuning**: With weight_decay=1e-3 (needed for strong head regularization), Adam's effective decay is significantly weaker than intended because ImageNet-pretrained backbone gradients tend to be small (the features are already close to their optimal values). AdamW applies the full 1e-3 decay regardless.
>
> **Practical impact**: One-line change — `from torch.optim import AdamW`. Same hyperparameters, correct regularization behavior. Difference is most noticeable with weight_decay > 1e-4.

**Q26d. What is differential learning rate in transfer learning? How did you implement it?**

> **Concept**: Different parts of a pretrained network need different learning rates during fine-tuning:
> - **Backbone** (pretrained features): already near-optimal for general vision features. Use a very small LR (1e-5) to gently adapt to new domain without overwriting learned structure.
> - **Classifier head**: randomly initialized, needs to learn from scratch. Use a larger LR (1e-4) to converge quickly.
>
> **Implementation** with AdamW param groups:
> ```python
> backbone_params   = [p for n, p in model.named_parameters()
>                      if p.requires_grad and "classifier" not in n]
> classifier_params = [p for n, p in model.named_parameters()
>                      if p.requires_grad and "classifier" in n]
>
> optimizer = AdamW([
>     {"params": backbone_params,   "lr": 1e-5, "weight_decay": 1e-5},
>     {"params": classifier_params, "lr": 1e-4, "weight_decay": 1e-3},
> ])
> ```
> - Backbone weight_decay=1e-5: very light — heavy L2 pushes pretrained weights toward zero, destroying ImageNet features.
> - Head weight_decay=1e-3: strong regularization — the linear classifier must generalize across 22 classes at 140 samples/class.
>
> **When to use**: Any time you unfreeze pretrained backbone layers for fine-tuning. Essential for Round 3 MobileNetV2 and ResNet50.

**Q27. What is the inference benchmark? Why 10 warmup runs?**

> `benchmark_inference()` measures average time for a single forward pass:
> ```python
> dummy = torch.randn(1, 3, 224, 224).to(device)
> for _ in range(10):    # Warmup — GPU cache cold start
>     _ = model(dummy)
> start = time.perf_counter()
> for _ in range(100):   # Measurement — 100 runs
>     _ = model(dummy)
> elapsed_ms = (time.perf_counter() - start) * 1000 / 100
> ```
> **Why 10 warmup runs**: The first few GPU forward passes are slower because: (1) CUDA kernel compilation happens on first call (JIT compilation), (2) GPU memory caches are cold. Warmup runs get the GPU into steady state. Measuring without warmup gives inflated (pessimistic) latency numbers. The 100-run average smooths out variance.

---

## Section 6 — MLflow & MLOps Questions

**Q28. What is MLflow? What did you use it for?**

> MLflow is an open-source platform for managing the ML lifecycle. In SmartVision AI, I used it for experiment tracking in Section 7: every training run for all 4 CNNs and YOLOv8n is logged to two experiments — `smartvision_classification` (4 runs) and `smartvision_detection` (1 run) — in a local SQLite database with WAL mode enabled.
>
> **Per CNN run, logged:**
> - Params: model name, epochs, dataset_round, fast_mode, num_classes
> - Metrics: test_accuracy, test_precision, test_recall, test_f1, val_accuracy, model_size_mb, cpu_inference_ms
> - Tags: weights_available, metrics_complete (False for VGG16 — its precision/recall/F1 were not measured)
> - Artifacts: confusion_matrix.png, training_history.png, metrics.json (if available)
>
> **Design decision:** VGG16's precision/recall/F1 are null — they were not measured during training and cannot be reliably derived from accuracy on a 22-class unbalanced problem. Rather than fabricate values, I set them to null and tagged the run `metrics_complete=False`. MLflow filters `if v is not None` before logging, so the VGG16 run has only accuracy and size metrics. This is cleaner than logging fabricated numbers that would corrupt any downstream champion-challenger comparison.

**Q28a. How did you handle MLflow concurrency — can you view results while the notebook is still running?**

> SQLite in default journal mode takes an exclusive write lock, so `mlflow ui` (which reads the DB) would block while the logging loop is writing. I enabled WAL (Write-Ahead Log) mode with an explicit `PRAGMA journal_mode=WAL; COMMIT;` before MLflow opens the file. WAL allows concurrent reads and writes: the UI can query while the notebook is logging new runs without either blocking. I also used an absolute path for the SQLite URI to avoid CWD-relative resolution issues (`(PROJECT_ROOT / "mlruns" / "mlflow.db").resolve()`).

**Q28b. How do you make MLflow logging resilient to partial failures?**

> Three patterns: (1) `try/except` per model run — if EfficientNet's logging crashes (missing artifact, SQLite lock), VGG16 and MobileNet runs are already committed. (2) `run_ids.json` is written incrementally after every run — if the process is killed after MobileNet's run, its run ID is already on disk. (3) `log_artifact_if_exists()` helper — missing PNG files become tags (`missing_confusion_matrix=path`) instead of exceptions. The loop completes all models regardless of which artifacts are absent.

**Q29. Why SQLite backend for MLflow instead of the default file store?**

> MLflow's default file store writes one JSON file per metric/parameter per run. With 4 models × multiple epochs × multiple metrics, that's hundreds of tiny files — `mlflow.search_runs()` has to scan a directory tree to answer queries, which is slow. SQLite stores everything in one `.db` file and supports proper SQL queries. `mlflow.search_runs()` is significantly faster. It also avoids Windows path issues with the default store. Trade-off: SQLite is single-writer per transaction (mitigated with WAL mode for concurrent reads). A team would use PostgreSQL or MySQL.

**Q29a. Why use `experiment_id` instead of `experiment_name` when searching MLflow runs?**

> `mlflow.search_runs(experiment_ids=[...])` is the safe API. `experiment_names=` was added later and in some versions silently returns an empty DataFrame if the name contains special characters or was renamed. Experiment IDs are stable integers assigned at creation — they never change. I retrieve the ID with `mlflow.set_experiment().experiment_id` immediately after creating/finding the experiment, then use that ID for all subsequent calls. This is especially important in automated pipelines where a silent empty search result looks like "no runs found" instead of an error.

**Q30. Why is there an 80% accuracy threshold before uploading to HuggingFace Hub?**

> It's a quality gate that prevents shipping broken models. If test accuracy is 78%, something went wrong:
> - Data corruption (a class has too few samples)
> - Hyperparameter bug (wrong LR, wrong batch size)
> - Wrong architecture (classifier head shape mismatch)
> - Training didn't converge (too few epochs)
>
> The assertion `assert test_metrics["accuracy"] >= 0.80` stops the upload and forces investigation. Without this gate, a broken model might get uploaded to HuggingFace, downloaded by the Docker container, and silently serve wrong predictions in production.

**Q31. Why store model weights on HuggingFace Hub instead of Git LFS?**

> VGG16 weights are ~550MB. Git LFS has a 2GB per-file bandwidth limit on the free tier and requires per-repo setup. HuggingFace Hub is purpose-built for ML artifacts: (1) Free storage for model weights, (2) `hf_hub_download()` at inference time gives the serving container exactly the weights it needs, (3) Versioned — each upload gets a commit hash, so you can pin the Docker container to a specific model version, (4) No Git LFS billing concerns.

---

## Section 7 — Architecture & System Design Questions

**Q32. Explain your two-phase architecture. Why not train on the server and serve from the same container?**

> **Phase A (Google Colab T4):** Stream COCO → preprocess → train 4 CNNs → train YOLOv8 → verify (≥80%) → upload weights to HuggingFace Hub → commit artifacts to git.
>
> **Phase B (Docker Compose):** FastAPI lifespan downloads weights from HF Hub → loads models into memory → warms up → serves `/classify`, `/detect`, `/metrics`. Redis caches results. Prometheus scrapes. Grafana dashboards.
>
> **Why separate**: (1) GPU cost — Colab provides free T4 GPU for training; a production server with a T4 costs $400+/month. (2) Training is one-time and slow; serving is continuous and must be low-latency. (3) Decoupling means the model weights are versioned artifacts — you can update them without rebuilding the entire container. (4) The serve container doesn't need PyTorch training code, scikit-learn, or jupyter — smaller Docker image, smaller attack surface.

**Q33. Streamlit never loads models directly. Why?**

> Three reasons: (1) **Separation of concerns** — Streamlit is a display layer, not a compute layer. If it loaded models, it would need 550MB+ of GPU memory per user session. (2) **Scalability** — FastAPI can be scaled horizontally (multiple replicas); Streamlit cannot share model instances across replicas. (3) **Security** — models expose an inference API, not raw weight files. Streamlit only calls `requests.post(f"{FASTAPI_URL}/classify", ...)` and renders the response — it never touches model weights.

**Q34. Why FastAPI lifespan context manager instead of `@app.on_event("startup")`?**

> `@app.on_event` is deprecated in FastAPI 0.93+ and will be removed in a future version. The lifespan context manager is the current recommended approach:
>
> ```python
> @asynccontextmanager
> async def lifespan(app: FastAPI):
>     # Startup: load models, connect Redis
>     models = load_all_models()
>     redis = connect_redis()
>     yield
>     # Shutdown: clean up
>     redis.close()
> ```
>
> The `yield` separates startup from shutdown cleanly, and it's more Pythonic than two separate decorated functions. It also integrates better with dependency injection.

**Q35. How does Redis caching work? What happens on a cache miss?**

> When a classification request arrives with an image:
> 1. Compute key: `sv:classify:{SHA256_32chars}:{model_name}:{model_hash_8chars}`
> 2. HIT: `pop("cached", None)` from stored dict, return `ClassifyResponse(**data, cached=True)`
> 3. MISS: run inference, `pop("cached")` before storing, `cache.set(key, payload, TTL)`
>
> **Why pop "cached" before storing?** `result.model_dump()` includes `cached=False`. Unpacking `**cached_data, cached=True` passes `cached` twice — Python raises `TypeError: keyword argument repeated` on every cache hit. Fix: store without the `cached` field; inject it on retrieval.
>
> **TTLs**: Classification: 86,400s (24h). Detection: 3,600s (1h) — scenes change faster.
>
> **Model hash in key**: when model weights update, hash changes, old cache entries become unreachable automatically — no explicit flush needed.
>
> **Graceful degradation**: `socket_connect_timeout=1.0` — Redis unavailable fails in 1s (not 20-30s OS timeout), sets `_available=False`, all get/set are no-ops. API continues serving. Rule 23.

**Q35a. Why asyncio.get_running_loop() and not get_event_loop() in FastAPI lifespan?**

> FastAPI lifespan is an `async` function — the event loop is already running when it executes. `get_event_loop()` raises DeprecationWarning in Python 3.10 and RuntimeError in Python 3.12+.
>
> ```python
> loop = asyncio.get_running_loop()   # correct
> models, hashes = await loop.run_in_executor(None, lambda: load_all_models())
> ```
>
> `run_in_executor` runs blocking `torch.load` + HF Hub download in a thread pool — event loop stays responsive so `/health` can return 503 during startup and handle SIGTERM for graceful shutdown.

**Q35b. Why UploadFile = File(...) not Form(...) for image bytes in FastAPI?**

> `Form()` is for text fields. Using `Form` for binary image data causes `422 Unprocessable Entity` on every request — no helpful error message.
>
> ```python
> # CORRECT: UploadFile for binary, Form for text fields
> async def classify(file: UploadFile = File(...), model_name: str = Form("resnet50")):
>     image_bytes = await file.read()
> ```
>
> Client: `files={"file": ("img.jpg", f, "image/jpeg")}, data={"model_name": "resnet50"}`.

**Q36. What is your Docker memory limit and why?**

> `mem_limit: 1.5g` in docker-compose.yml. Breakdown: VGG16 weights = 550MB, ResNet50 = 100MB, MobileNetV2 = 14MB, EfficientNetB0 = 20MB, YOLOv8n = 6MB → total ~690MB for all models loaded. The 1.5GB limit gives ~810MB headroom for inference activation memory, PyTorch runtime, FastAPI worker threads, and concurrent requests. Without a limit, a memory leak or runaway inference could consume the host machine's RAM.

**Q36b. Your Streamlit API client catches `ConnectionError` — what happens if FastAPI returns a 503?**

> `requests` does not raise on non-200 responses by default — it returns the response object. Without `raise_for_status()`, a 503 (models loading) silently returns a JSON body with `{"detail": "Models not yet loaded"}`. The caller tries `resp.json()["predictions"]` and gets a `KeyError` — a raw Python traceback in Streamlit instead of a user-friendly error message.
>
> Fix: every `api_client` function calls `resp.raise_for_status()` inside a `try/except Exception` block that feeds into a central `_handle_error()` function. The final branch is `raise RuntimeError(f"Unexpected error: {type(e).__name__}: {e}")` — this covers `JSONDecodeError` from HTML proxy error pages, `ChunkedEncodingError` from dropped connections, and any other type not explicitly caught. Pages only need `try/except RuntimeError → st.error(str(e))`.

**Q36c. How do you prevent the Streamlit Drift Monitor page from losing its data on every UI interaction?**

> Streamlit re-runs the entire page script on every interaction — a slider change, a selectbox update, even scrolling. Without `st.session_state`, data fetched by a "Refresh" button is lost on the next interaction and the page appears blank.
>
> Fix: `st.session_state["drift_status"] = api_client.get_drift_status()` on button click. Render from `st.session_state.get("drift_status")` unconditionally. Data persists within the session, making the Refresh button a deliberate explicit trigger — not accidentally re-triggered by UI interaction. This also avoids auto-polling loops (`st.rerun()` + `time.sleep()`) which create runaway re-render cycles.

**Q36d. A user uploads a photo taken in portrait mode on their phone. Your bounding boxes appear in the wrong positions. Why? How do you fix it?**

> JPEG photos from phone cameras encode orientation in EXIF metadata (e.g., tag 274 = 6 means "rotated 90° CW"). PIL's `Image.open()` does NOT apply EXIF rotation by default — it returns the raw pixel data as stored, which is landscape-orientation for a portrait photo.
>
> FastAPI receives the raw bytes, opens with PIL (wrong orientation), passes to YOLO → YOLO returns bboxes in the unrotated coordinate system. Streamlit displays the image (also without EXIF correction), draws bboxes → they land in the correct positions on the wrong-orientation image. The user sees a rotated photo with boxes in strange positions.
>
> Fix: `ImageOps.exif_transpose(pil_image)` must be called on both sides:
> - **API** (`detect.py`): before converting to numpy array for YOLO — so bboxes are in the corrected coordinate system.
> - **Streamlit** (`2_Detect.py`): before PIL drawing — so the display image matches the coordinate system YOLO used.
>
> Both sides must apply the same transformation, or the coordinate systems diverge.

---

## Section 8 — Monitoring & Drift Detection Questions

**Q37. What is model drift? What kind of drift are you detecting?**

> Model drift is when the real-world data distribution shifts away from what the model was trained on, causing accuracy to degrade silently without any errors being thrown. There are two types: (1) **Data drift** — input images change (e.g., new camera angle, new lighting conditions). (2) **Concept drift** — the relationship between inputs and correct labels changes.
>
> We detect **confidence score drift** using the Kolmogorov-Smirnov (KS) test. We save a baseline distribution of confidence scores from the validation set (held-out, not seen during training). At serving time, we accumulate confidence scores from real requests and periodically compare them to the baseline using KS. If the distribution has shifted significantly, we flag a drift alert.

**Q37a. Why use the validation split for the drift baseline — not the training split?**

> The model was trained on training images — it has seen those images during gradient updates and will produce systematically inflated confidence scores on them (memorization effect). A baseline built from training-split confidence reflects the model's "familiar territory" distribution, not the distribution it would see in deployment.
>
> If the baseline uses training confidence (higher mean, tighter distribution) and live traffic confidence is lower (as expected for novel images), the KS test will flag this gap as drift — even when the model is behaving normally. This is a permanent source of false positive alerts from the first day of deployment.
>
> The validation split is held-out: the model never saw these images during training. Its confidence on val images represents what you'd expect on new deployment traffic. Val split was 30 images/class = 660 total — sufficient for a two-sample KS test (which has power at n=30).

**Q37b. You mentioned your baseline model (MobileNet, 56.7%) is overconfident on errors. How did you handle that?**

> Max-softmax from a model with below-70% accuracy frequently produces high-confidence wrong predictions — the model's uncertainty is not well-calibrated. A single pooled confidence distribution mixes high-confidence-correct and high-confidence-incorrect samples, obscuring what's actually happening.
>
> I added accuracy-conditional distributions to the baseline: for each class, I record `mean_correct` (average confidence when the model predicts correctly) and `mean_incorrect` (when it predicts wrongly), alongside the raw mean. Section 9's drift detector can then flag: "live confidence is falling below `mean_correct` for class airplane" — which is a more specific and actionable signal than "the aggregate distribution shifted." This also documents the known limitation of using a 56.7% model for the baseline, with a note to recompute after Section 12 uploads ResNet50 (65.5%) to HuggingFace Hub.

**Q38. Why KS test on confidence scores, not on raw image embeddings?**

> Embedding-based drift requires storing high-dimensional vectors (2048-dim for ResNet50) for every prediction — expensive in storage and computation. Confidence scores are scalar (one float per prediction) — cheap to store, cheap to compare. Practically: if a model is seeing classes it wasn't trained on, or if image quality has degraded, the model will produce lower or more diffuse confidence distributions — the KS test will catch this. Rule 21: KS on confidence, not embeddings.

**Q39. What does a KS threshold of 0.10 mean?**

> The KS statistic measures the maximum absolute difference between two CDFs (cumulative distribution functions) — the baseline distribution and the current serving distribution. A KS statistic of 0.10 means the distributions differ by at most 10% at any confidence level. Our alert threshold: if KS > 0.10, trigger a drift alert. The minimum sample requirement of 100 predictions ensures the KS test has enough data to be statistically meaningful — a KS test on 5 predictions is noisy and unreliable.

**Q40. What metrics does Prometheus scrape? What does Grafana dashboard?**

> **Prometheus-exposed metrics (via `/metrics` endpoint, text/plain format)**:
> - Request count per endpoint (classify, detect)
> - Inference latency histogram (p50, p95, p99)
> - Prediction distribution by class (label: `class_name`)
> - Cache hit/miss ratio
> - KS drift statistic, p-value, and alert flag per class (Section 9)
> - Live buffer size per class (how many inferences have been collected)
>
> **Grafana dashboards**: Real-time request rate, latency trends, class prediction frequency (heatmap), drift score time series per class, Redis cache efficiency. The consistent label name `class_name` across all Prometheus metrics means a single Grafana variable can filter by class across all panels.

**Q40a. Your KS threshold is 0.10 — why not just use `stat > 0.10` as the alert condition?**

> I tried that first. `stat > 0.10` alone fires on almost every class even when there's no real drift. The KS D-statistic has high variance at small reference sample sizes — our baseline is only 30 samples per class (from the val split). With n_baseline=30 and n_live=100, sampling noise alone produces KS stats of 0.11-0.25 regularly. You'd be paged constantly with no real signal.
>
> The fix: double-gate alert (`stat > 0.10 AND p-value < 0.05`). The p-value accounts for the sample size — it stays high (non-significant) when the stat is elevated by chance at small n. In practice: stat=0.22, p=0.12 → no alert (sampling noise). stat=0.60, p=0.000001 → alert fires (real shift of +0.3 mean).
>
> Verified empirically: Test B (same distribution, n=110) → stat=0.02, p=1.00. Test C (+0.3 shift, n=150) → stat=0.60-0.82, p≈0. Zero false positives, 100% detection.

**Q40b. Why do you rate-limit the KS test instead of running it on every request?**

> The KS test itself is fast (~0.5ms for n=200 vs n=30). But classify requests arrive at up to 100/s. With 22 classes and KS running on every `record()` call after the buffer fills, that's 2,200 KS executions per second → 1.1 seconds of CPU blocked per second. Since `record()` runs in the asyncio event loop thread (it's synchronous, no await), this directly delays the next HTTP request's response time.
>
> Fix: `KS_RUN_EVERY_N=10` — run KS every 10th new sample per class. Reduces event loop blocking from 1.1s/s to 110ms/s. Statistically equivalent: 10 consecutive inferences from the same request pattern provide zero additional information for the KS test.

**Q40c. How does your DriftDetector survive an API restart without losing collected data?**

> Live confidence scores are dual-written: to an in-memory `deque(maxlen=200)` for fast access, and to a Redis list `sv:drift:live:{class_name}` (LPUSH + LTRIM to 200, via pipeline) for persistence.
>
> On restart, `DriftDetector.__init__` calls `redis_client.get_list(key)` for each class, which returns the 200 most recent scores in newest-first order (LPUSH order). We reverse the list before `deque.extend()` to restore chronological order — without reversing, the deque would evict the most recently collected samples first on next insert, which is backwards.
>
> One important note: the LPUSH+LTRIM pipeline is NOT atomic (not MULTI/EXEC). With a single uvicorn worker (our setup), there's no race condition. With multiple workers, the list could transiently exceed 200 items. Documented in the code; acceptable for this deployment model.

---

## Section 9 — Python & Coding Questions

**Q41. You wrote a NumpyEncoder. Why is it needed? Write it.**

> PyTorch and numpy return native typed objects — `np.float64`, `np.int64`, `torch.Tensor` — that Python's built-in `json.dumps()` cannot serialize. Without a custom encoder, saving metrics to JSON raises `TypeError: Object of type float64 is not JSON serializable`.

```python
import json
import numpy as np
import torch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        if isinstance(obj, np.bool_):     # MUST be before np.integer
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
```

> **Critical**: `np.bool_` must be checked BEFORE `np.integer`. In numpy < 2.0, `np.bool_` is a subclass of `np.integer` — checking integer first converts `True` to `1` (int) instead of `True` (bool). The JSON and downstream consumers expect booleans for fast_mode, arrest_binary, etc.

**Q42. Why `torch.save(model.state_dict(), path)` and not `torch.save(model, path)`?**

> Saving the full model with `torch.save(model, path)` uses Python pickle, which serializes the entire class definition, not just the weights. This causes two problems: (1) **Security**: `torch.load()` of a full pickled model executes arbitrary Python code — a malicious weights file could run any code on the loading machine. (2) **Portability**: if the model class is refactored (renamed, moved to a different module), the saved file becomes unloadable. Saving only `state_dict()` saves just the tensor weights, which load cleanly with `weights_only=True`:
>
> ```python
> # Save
> torch.save(model.state_dict(), save_path)
>
> # Load (Rule 36)
> model = get_model(name, num_classes=25)     # Reconstruct architecture first
> state = torch.load(save_path, weights_only=True)
> model.load_state_dict(state)
> ```

**Q43. Why `asyncio.get_running_loop()` instead of `asyncio.get_event_loop()`?**

> `get_event_loop()` is deprecated in Python 3.10+ and will raise `DeprecationWarning` if called outside an async context. In Python 3.12, it raises `RuntimeError` if there is no current event loop. `get_running_loop()` is explicit — it returns the currently running event loop or raises `RuntimeError` if called outside one, making bugs loud and clear. In FastAPI's async context (inside lifespan or route handlers), there is always a running loop — `get_running_loop()` is the correct, forward-compatible call.

**Q44. Write the checkpoint/resume pattern for data acquisition.**

```python
def load_checkpoint() -> dict:
    """Returns dict[class_name → count] or empty dict if no checkpoint."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}

def save_checkpoint(progress: dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(progress, indent=2))

# In data acquisition:
progress = load_checkpoint()
completed = {cls for cls, cnt in progress.items() if cnt >= target}
remaining = [cls for cls in CLASSES if cls not in completed]

if remaining:
    for item in stream:
        for cat_id in set(item["objects"]["category"]):
            if cat_id not in HF_CATEGORY_TO_CLASS_IDX:
                continue
            class_name = CLASSES[HF_CATEGORY_TO_CLASS_IDX[cat_id]]
            if class_name not in remaining:
                continue
            class_images[class_name].append(item)
            class_counts[class_name] += 1
            if class_counts[class_name] >= target:
                remaining.remove(class_name)
                save_checkpoint(class_counts)  # Persist after each class completes
```

---

## Section 10 — Evaluation & Metrics Questions

**Q45. Why macro-averaged F1, not weighted F1 or accuracy?**

> With 25 classes, each class gets equal weight in macro-averaged F1, regardless of how many samples it has. This matters because:
> - If 'person' has 1000 test samples and 'bed' has 15, weighted F1 would give 'person' 66× more weight — effectively ignoring how well the model performs on rare classes.
> - Macro F1 says: "how well does the model perform on EACH class on average?" — the right question for a balanced class system.
> - We use 70/15/15 split with exactly 100 images per class, so the dataset is balanced — but macro still prevents any future imbalance from silently dominating the metric.

**Q46. Which classes are hardest to classify? How do you know?**

> The EDA class difficulty prediction (from `02_eda.py`) identifies visually similar class pairs:
> - **Hard (≥2 similar neighbours + high size similarity):** car ↔ truck ↔ bus (all wheeled vehicles, overlapping sizes)
> - **Medium (1 similar neighbour):** cat ↔ dog, cup ↔ bowl, couch ↔ chair
> - **Easy (no similar neighbours):** airplane, elephant, pizza (visually distinct shapes)
>
> This is validated against the actual confusion matrix after training. If confusion matrix shows car↔truck mix-ups and the EDA predicted it, the prediction is confirmed. If they disagree, the model learned more discriminative features than expected.

**Q47. What does your post-training verification step check?**

> Four checks, in order:
> 1. Model file exists and has reasonable size (not 0 bytes = incomplete save)
> 2. Model loads cleanly under `weights_only=True` and runs inference on a dummy input — output shape is `(1, 25)` and softmax probabilities sum to 1.0
> 3. Test accuracy ≥ 80% — assertion that blocks HF upload if not met
> 4. HuggingFace Hub upload succeeds and returns the uploaded path
>
> Only if all 4 pass is the model considered "done" for that architecture. Then commit artifacts, move to the next model.

---

## Section 11 — Behavioral / STAR Questions

**Q48. Describe the biggest technical challenge in this project and how you solved it.**

> **Situation**: VGG16 training completed with 59.2% val accuracy — well below the 80% threshold. This blocked the Section 5 pipeline.
>
> **Task**: Diagnose, fix, and decide whether further investment in VGG16 was justified.
>
> **Action — Round 1 (head-only diagnosis)**: Examined trainable param count: `102,425 / 134,362,969 = 0.1%`. Only the new head was training. ImageNet features weren't representing 25 COCO classes well enough for the head alone to compensate. Fix: implement 2-phase training. Added `freeze_vgg16_phase1()` and `unfreeze_vgg16_phase2()` to `model_factory.py`, updated training blocks in `.py` and `.ipynb`, added param count printouts between phases.
>
> **Action — Round 2 (Phase 2 diagnosis)**: Phase 2 ran correctly — trainable params confirmed at 7,181,849 before training started. But the training curve showed a problem: train accuracy climbed from 58% → 92% across 15 epochs while val accuracy plateaued at 59.5%. A **33% train-val gap** is textbook overfitting. Calculated the root cause: `7,100,000 params / 1,725 training images = 4,100 params per image`. At that ratio, the model has enough capacity to memorise training data rather than learn from it.
>
> **Decision**: Accept VGG16 at 59.5% and document it as an architecture limitation. Move to ResNet50 immediately. This was a deliberate call — VGG16's last block has 3–7× more parameters than modern architectures (ResNet50: ~2.1M, MobileNetV2: ~300K). It's not designed for fine-tuning on 1,725 images.
>
> **Result — VGG16**: Wrote in the experiment log: "VGG16: 59.5% val. Overfitting Phase 2 (train 92% / val 59%, 33% gap). 4,100 params/image. Architecture limitation — moving to ResNet50."
>
> **Action — Round 3 (ResNet50 same pattern, larger scale)**: ResNet50 Phase 2 (layer3+layer4 unfrozen = 24M params) repeated the overfitting: train 99%, val 70%, 29% gap. Calculated: `24,000,000 / 1,725 = 13,913 params/image` — 3× worse than VGG16. Fix: changed `unfreeze_resnet50_phase2()` to unfreeze **only layer4** (~16M params, 9,275 params/image). This is a 1-line change — removed "layer3" from the unfreezing condition. Documented the reason in the function docstring so any future developer understands why layer3 stays frozen.
>
> **The pattern I established**: params/image > ~5,000 → overfitting risk high → reduce unfrozen layers. This is now a repeatable diagnostic rule across all 4 models.

**Q49. How did you ensure your results are reproducible?**

> Three practices: (1) `RANDOM_STATE = 42` set in `config.py` and passed to `torch.manual_seed()`, ensuring weight initialisation and dataloader shuffling are deterministic. (2) All hyperparameters live in `config.py` only — no magic numbers inline in notebook cells. Any change is in one place, visible in git history. (3) MLflow logs every run's exact parameters and timestamp. To reproduce a run from last week: check git log for the config at that commit, read MLflow params for that run, re-execute — identical output guaranteed.

**Q50. You found bugs in your mentor's reference notebook. Describe them and how you handled it.**

> **Bug 1 — 26 classes instead of 25:** The mentor's notebook included 'train' (the vehicle, not the train/val split) as class 6, giving 26 classes. Their YAML had `nc: 26`. The project PDF specifies 25. I detected this by counting `SELECTED_CLASSES` keys vs the PDF spec. Our `verify_coco_mapping()` catches this in one assertion.
>
> **Bug 2 — Flat detection folder:** The mentor dumped all train+val detection images into a single `images/` directory with no train/val split. YOLOv8 training requires `images/train/` and `images/val/` separately. Training on a flat folder would either fail or produce a model with no validation monitoring. We maintain a proper split in `detection/images/train/` and `detection/images/val/`.
>
> I documented both bugs in the notebook header (`01_data_acquisition.py` lines 7–9) so the improvements are visible and explainable.

---

## Section 12 — Advanced / MNC Questions (Google India, Microsoft India, Amazon)

**Q51. How would you scale SmartVision AI to 100 classes?**

> Five changes: (1) **Data**: 100 images/class × 100 classes = 10,000 images — streaming becomes more important (more iterations to fill all classes). Checkpoint/resume already handles this. (2) **Memory**: the final linear layer grows from 4096→25 to 4096→100 — negligible weight increase. (3) **Architecture**: with 100 classes, more training data per class is needed to avoid overfitting. MobileNetV2/EfficientNetB0 may need fine-tuning (currently frozen). (4) **Class imbalance**: harder to maintain perfect balance across 100 classes from COCO — need weighted sampling or focal loss. (5) **Inference**: `/classify` response grows from 25 scores to 100 — still fast, but Streamlit display needs redesign (not all 100 bars fit on screen).

**Q52. How would you serve these models on mobile/edge devices?**

> MobileNetV2 (14MB) and EfficientNetB0 (20MB) are the candidates — designed for efficient inference. Steps: (1) **Quantize**: `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` reduces MobileNetV2 from 14MB to ~3.5MB and inference latency by 2–4×. (2) **Export to TorchScript or ONNX**: `torch.onnx.export(model, dummy_input, "model.onnx")` for cross-platform deployment. ONNX Runtime runs on Android/iOS. (3) **TFLite conversion**: for Android, convert ONNX → TFLite for the most efficient mobile inference. VGG16 (550MB) is not a mobile candidate — it doesn't fit in most phone RAM budgets.

**Q53. How would you add a new class without retraining all 4 models from scratch?**

> Two approaches depending on how different the new class is:
>
> **If new class is visually similar to existing ones (e.g., adding 'van' near 'car/truck'):** Incremental fine-tuning — extend the output head from 25→26 neurons, freeze all conv layers, train only the head on new class data + a 20% sample of old classes to prevent forgetting. This takes a fraction of original training time.
>
> **If new class is very different (e.g., adding 'submarine'):** Full 2-phase retraining with all 26 classes. The existing weights are a good starting point — Phase 1 and Phase 2 will converge faster than training from scratch.
>
> In both cases: version the new model separately in MLflow, upload to a new HF Hub path, update the Docker config — don't overwrite the production model until the new one passes 80% threshold.

**Q54. Real-time vs batch inference — which does SmartVision AI use and why?**

> SmartVision AI uses **online (real-time) inference**: each request triggers a single forward pass and returns results immediately. This suits the use case — interactive demo where a user uploads an image and expects a response in <100ms.
>
> **Batch inference** would be better for: processing 10,000 product images overnight for a retail catalogue update. Batch inference on GPU is 20–50× more efficient (tensor parallelism, memory bandwidth utilisation). Our FastAPI architecture could support batch inference by accepting a list of images per request and running `model(batch_tensor)` — we intentionally kept the API simple for the demo.
>
> **Redis caching bridges the gap**: repeated requests for the same image (common in demos) get cached results in <1ms — effectively free.

---

## Section 13 — Quick-Fire Conceptual Questions

**Q55.** What is transfer learning?
> Reusing weights from a model trained on a large dataset (ImageNet, 1.28M images) as a starting point for a new, related task. Early layers learn general features (edges, textures) reusable across tasks. Only the final head needs task-specific training.

**Q56.** What are ImageNet normalization mean and std values? Why do they matter?
> Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] (RGB channels). All torchvision pretrained models were trained with these exact statistics applied to input images. Using different normalization shifts all activations in the network — features learned from ImageNet no longer activate correctly on your input. Must apply **at both train and inference time** to avoid train/serve skew.

**Q57.** What is the difference between `state_dict` and a full pickled model?
> `state_dict`: a Python dict mapping layer names to parameter tensors — just the weights, no code. Safe to load with `weights_only=True`. Full model pickle: the entire class definition + weights — can execute arbitrary code on load (security risk) and breaks if the class is renamed.

**Q58.** What is a confusion matrix? What does the off-diagonal tell you?
> A K×K matrix where entry (i,j) = number of samples of true class i predicted as class j. The diagonal = correct predictions. Off-diagonal entries = errors: row = true class, column = predicted class. A high value at (car, truck) means many car images were predicted as truck — a class confusion problem.

**Q59.** What is GradScaler and when do you use it?
> `torch.cuda.amp.GradScaler` is used with mixed precision training. In float16, gradients can underflow to zero (values too small to represent in float16). GradScaler multiplies the loss by a large scale factor before backward pass, keeping gradients in float16 range, then divides the gradients back before the optimizer step. Used only with `torch.autocast("cuda")` — not needed for float32 training.

**Q60.** What is `@st.cache_data` vs `@st.cache_resource`?
> `@st.cache_data`: caches the function's return VALUE — used for loading CSVs, JSONs, dataframes. Streamlit serializes the return value; different inputs → different cache entries. `@st.cache_resource`: caches the OBJECT INSTANCE — used for model objects, database connections. The object is shared across all users and sessions; don't use for data that changes per user.

**Q61.** What is `weights_only=True` in `torch.load()`?
> A security parameter added in PyTorch 1.13. Without it, `torch.load()` uses full Python unpickling which can execute arbitrary code embedded in the weights file — a malicious .pt file could be a trojan. `weights_only=True` restricts loading to only tensor data, preventing code execution. Will become the default in a future PyTorch version.

**Q62.** What is the difference between model.eval() and torch.no_grad()?
> `model.eval()`: sets the model to evaluation mode — disables dropout layers and makes BatchNorm use running statistics instead of batch statistics. Does NOT stop gradient computation. `torch.no_grad()`: disables gradient tracking for all operations in the context — saves memory and speeds up inference. For inference, you need BOTH: `model.eval()` for correct behaviour, `torch.no_grad()` for efficiency.

---

## Section 14 — Preparation Tips for India's Big Tech Interviews

**Service companies (HCL/TCS/Wipro/Infosys):**
- Expect a 45-min project walkthrough. Have a 2-min intro (Q1), a 5-min deep dive on one model's training (Q14 — VGG16 2-phase), and a 3-min system overview (Q32) ready.
- They will ask "What is your contribution vs. what tools did?" — be clear that you designed all training strategies, debugged the 59.2% problem, found the mentor's 26-class bug, and wrote all the code.
- Know your numbers cold: 2500 images, 25 classes, 70/15/15, 80% threshold.

**Product companies (Flipkart/Amazon/Zomato):**
- Expect a system design question: "Design a real-time product image classification system." Use Q32 and Q33 as your answer — Colab for training, FastAPI for serving, Redis for caching, Streamlit for display.
- They will ask "what would you do differently?" → Q53 (incremental learning for new classes) and Q51 (scaling to 100 classes).
- Expect trade-off questions: "real-time vs batch inference" → Q54.

**FAANG-adjacent (Google/Microsoft India):**
- Expect deep theory probes: "Derive the cross-entropy loss for multi-class classification." "Why does Adam work better than SGD for transfer learning?" (Q26).
- Expect ML system design: model versioning (MLflow + HF Hub), serving architecture (FastAPI + Redis), monitoring (Prometheus + KS drift).
- Be ready to discuss what's NOT in the project: "How would you add A/B testing for model updates?"

---

## Metrics to Memorize

| Fact | Value |
|---|---|
| Dataset | 2,500 images, 25 COCO classes, 100/class |
| Split | 70% train (1,750) / 15% val (375) / 15% test (375) |
| FAST_MODE target | 10 images/class = 250 total |
| Streaming safety limit | MAX_ITER = 60,000 images |
| Image size (classification) | 224 × 224, ImageNet normalized |
| Image size (detection/YOLO) | 640 × 640 |
| ImageNet mean | [0.485, 0.456, 0.406] |
| ImageNet std | [0.229, 0.224, 0.225] |
| VGG16: batch/epochs/phases | 16 / 20 (5+15) / 2-phase |
| VGG16: trainable Phase 1 | ~102K / 134M (0.1%) |
| VGG16: trainable Phase 2 | ~7.2M / 134M (5.4%) |
| VGG16: Phase 1 val accuracy | 59.2% (head-only) |
| VGG16: Phase 2 val accuracy | 59.5% (overfit — train 92% / val 59%, 33% gap) |
| VGG16: params/image ratio | 4,100 (7.1M params / 1,725 images — too high) |
| VGG16: decision | Architecture limitation — accepted 59.5%, moved to ResNet50 |
| VGG16: model size | ~550MB |
| ResNet50: batch/epochs/phases | 32 / 25 (6+19) / 2-phase |
| ResNet50 Phase 2 (layer3+4, attempt 1) | 70.4% val (overfit — train 99% / val 70%, 29% gap) |
| ResNet50 params/image (layer3+4) | 13,913 — too high (24M / 1,725) |
| ResNet50 Phase 2 (layer4 only, attempt 2) | TBD — expected 76-82% |
| ResNet50 params/image (layer4 only) | 9,275 — safer (16M / 1,725) |
| ResNet50: model size | ~100MB |
| MobileNetV2: batch/epochs | 32 / 25 (10+15) / 2-phase |
| MobileNetV2 Round 2 test accuracy | 62.3% (below 65% floor — 27.4pp train/val gap) |
| MobileNetV2 Round 2 params/img | 552 (1.71M / 3,080) — features[14:] unfrozen |
| MobileNetV2 Round 3 params/img | 401 (1.23M / 3,080) — features[16:] unfrozen |
| MobileNetV2 Round 3 target | 68-74% (realistic for 200 samples/class) |
| MobileNetV2: model size | ~14MB (9.3MB saved checkpoint) |
| EfficientNetB0: batch/epochs | 32 / 25 / single-phase + AMP |
| EfficientNetB0: model size | ~20MB |
| YOLOv8: epochs/batch | 50 / 16 |
| YOLO conf threshold | 0.5 |
| YOLO IOU threshold | 0.45 |
| Accuracy gatekeeper | ≥ 80% before HF Hub upload |
| Redis TTL (classify) | 86,400s (24 hours) |
| Redis TTL (detect) | 3,600s (1 hour) |
| KS drift threshold | 0.10 (10% max CDF difference) |
| KS minimum samples | 100 predictions |
| FastAPI total memory | ~690MB (all models) |
| Docker mem_limit | 1.5g |
| MLflow backend | SQLite (mlruns/mlflow.db) |
| MLflow experiment | smartvision_classification |
| Inference benchmark | 10 warmup + 100 runs, mean ms |
| RANDOM_STATE | 42 (everywhere) |
