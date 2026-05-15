# Rule: One Model Per Session Turn

When implementing CNN training code, write ONE model's complete training block per session turn.

**Order:**
1. VGG16 — write + confirm it runs in Colab → STOP
2. ResNet50 — write + confirm → STOP
3. MobileNetV2 — write + confirm → STOP
4. EfficientNetB0 — write + confirm → STOP

After each model: run post-training verification, upload to HF Hub, commit artifacts.
ONLY THEN proceed to next model.

**Why:** EMI project crash — generating all 8 classifiers at once caused API failure and lost session.
