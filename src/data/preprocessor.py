# filename: src/data/preprocessor.py
# purpose:  YOLO annotation validation, data.yaml generation, dataset structure verification
# version:  1.0

import logging
from pathlib import Path

import yaml

from config import CLASSES, NUM_CLASSES, DATA_PROCESSED_DIR, ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def validate_yolo_annotations(labels_dir: Path) -> None:
    """
    Assert all YOLO label files have valid normalized values.
    Raises AssertionError immediately on the first invalid line.
    Call this before every YOLO training run.
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        logger.warning(f"Labels dir not found: {labels_dir}")
        return

    files_checked = 0
    lines_checked = 0

    for txt_file in labels_dir.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w   = float(parts[3])
            h   = float(parts[4])

            assert 0 <= class_id < NUM_CLASSES, (
                f"class_id {class_id} out of range [0, {NUM_CLASSES}) in {txt_file}"
            )
            # Use <= 1.0 — bbox_to_yolo clamps to min(1.0,...) so boundary values are valid
            assert 0.0 < x_c <= 1.0, f"x_center {x_c} not in (0,1] in {txt_file}"
            assert 0.0 < y_c <= 1.0, f"y_center {y_c} not in (0,1] in {txt_file}"
            assert 0.0 < w   <= 1.0, f"width {w} not in (0,1] in {txt_file}"
            assert 0.0 < h   <= 1.0, f"height {h} not in (0,1] in {txt_file}"
            # Catch clearly wrong values (negative or >1)
            assert x_c >= 0 and y_c >= 0, f"Negative center in {txt_file}"
            lines_checked += 1
        files_checked += 1

    print(f"YOLO annotations valid: {files_checked} files, {lines_checked} annotations in {labels_dir.name}/")


def create_yolo_data_yaml(detection_dir: Path) -> Path:
    """
    Generate data.yaml with absolute paths — works identically in terminal and Colab.
    Always regenerate before YOLO training (Colab paths differ from local paths).
    """
    detection_dir = Path(detection_dir)
    config = {
        "path":  str(detection_dir.absolute()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": {i: cls for i, cls in enumerate(CLASSES)},
    }
    yaml_path = detection_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Verify it loads cleanly
    with open(yaml_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["nc"] == NUM_CLASSES
    assert len(loaded["names"]) == NUM_CLASSES

    logger.info(f"data.yaml written: {yaml_path}")
    print(f"data.yaml OK: nc={loaded['nc']}, path={loaded['path']}")
    return yaml_path


def verify_dataset_structure() -> dict:
    """
    Count images in every split/class folder and return a summary dict.
    Confirms the dataset was built correctly before preprocessing.
    """
    from config import TRAIN_SPLIT, VAL_SPLIT

    cls_dir = DATA_PROCESSED_DIR / "classification"
    det_dir = DATA_PROCESSED_DIR / "detection"

    result: dict = {
        "classification": {},
        "detection": {},
        "issues": [],
    }

    # Classification counts
    for split in ["train", "val", "test"]:
        split_total = 0
        per_class: dict[str, int] = {}
        for cls in CLASSES:
            count = len(list((cls_dir / split / cls).glob("*.jpg"))) \
                    if (cls_dir / split / cls).exists() else 0
            per_class[cls] = count
            split_total += count
            if count == 0:
                result["issues"].append(f"classification/{split}/{cls}: 0 images")
        result["classification"][split] = {"total": split_total, "per_class": per_class}

    # Detection counts
    for split in ["train", "val"]:
        imgs = len(list((det_dir / "images" / split).glob("*.jpg"))) \
               if (det_dir / "images" / split).exists() else 0
        lbls = len(list((det_dir / "labels" / split).glob("*.txt"))) \
               if (det_dir / "labels" / split).exists() else 0
        result["detection"][split] = {"images": imgs, "labels": lbls}
        if imgs != lbls:
            result["issues"].append(
                f"detection/{split}: images({imgs}) != labels({lbls})"
            )

    # Expected totals
    total_cls = sum(
        result["classification"][s]["total"] for s in ["train", "val", "test"]
    )
    result["classification"]["total"] = total_cls
    result["detection"]["total_images"] = sum(
        result["detection"][s]["images"] for s in ["train", "val"]
    )

    return result
