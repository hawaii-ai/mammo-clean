import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "Skin", 1: "Clip", 2: "Wire"}


def load_yolo_class_ids(label_path: Path) -> list[int]:
    """Read a YOLO label file and return just the class IDs (boxes ignored).

    Args:
        label_path: Path to the YOLO-format label text file.
    """
    if not label_path.exists():
        return []
    classes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                classes.append(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def stem_from_tile(tile_path: Path) -> str:
    """Extract source image stem from a tile filename like 'abc__03.png'.

    Args:
        tile_path: Path to a tile image file.
    """
    stem = tile_path.stem
    return stem.rsplit("__", 1)[0] if "__" in stem else stem


def load_split_list(path: Path) -> list[str]:
    """Read a split list file and return stems (drops .png extension).

    Args:
        path: Path to the split list text file.
    """
    stems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".png"):
                line = line[:-4]
            stems.append(line)
    return stems


# Tiled mode

def load_exclusion_stems(path: Path) -> set:
    """Read a file with one filename per line and return the stems (no extension).

    Handles lines with or without a .png extension, and ignores blank lines
    and lines starting with '#'.
    """
    stems = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stem = line[:-4] if line.endswith(".png") else line
            stems.add(stem)
    return stems


def collect_tiled_inputs(images_dir: Path,
                         exclude_stems: set | None = None) -> list[tuple[Path, str]]:
    """Walk the tile dir and return (tile_path, source_stem) pairs.

    Args:
        images_dir: Directory of tile PNGs.
        exclude_stems: Source-image stems to drop entirely; build with load_exclusion_stems().
    """
    pairs = []
    for p in sorted(images_dir.glob("*.png")):
        src_stem = stem_from_tile(p)
        if exclude_stems and src_stem in exclude_stems:
            continue
        pairs.append((p, src_stem))
    return pairs


def aggregate_gt_tiled(pairs: list[tuple[Path, str]], labels_dir: Path) -> dict[str, set]:
    """Collect GT class IDs per source stem, unioning across all tiles.

    Args:
        pairs: List of (tile_path, source_stem) pairs from collect_tiled_inputs().
        labels_dir: Directory of tile YOLO label files.
    """
    gt_per_image = defaultdict(set)
    for tile_path, src_stem in pairs:
        lbl_path = labels_dir / f"{tile_path.stem}.txt"
        for c in load_yolo_class_ids(lbl_path):
            gt_per_image[src_stem].add(c)
        if src_stem not in gt_per_image:
            gt_per_image[src_stem] = set()
    return dict(gt_per_image)


def _apply_edge_pad(img: np.ndarray, edge_pad_px: int) -> np.ndarray:
    """Zero out the outer N-pixel ring before inference.

    Args:
        img: Grayscale image array to pad.
        edge_pad_px: Number of pixels to zero out around the border.
    """
    if edge_pad_px <= 0:
        return img
    out = img.copy()
    H, W = out.shape[:2]
    p = min(edge_pad_px, H // 2 - 1, W // 2 - 1)
    if p <= 0:
        return out
    out[:p, :] = 0
    out[-p:, :] = 0
    out[:, :p] = 0
    out[:, -p:] = 0
    return out


def _record_max_conf(preds_max_conf: dict, stem: str, box_iter) -> None:
    """Update max-confidence-per-class for a stem from an iterable of boxes.

    Args:
        preds_max_conf: Mapping of stem -> class_id -> running max confidence, updated in place.
        stem: Source image stem the boxes belong to.
        box_iter: Iterable of Ultralytics box objects.
    """
    for box in box_iter:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if conf > preds_max_conf[stem][cls_id]:
            preds_max_conf[stem][cls_id] = conf


def _apply_filters(boxes_obj, img: np.ndarray, filter_cfg, filter_module, total_rejects: dict) -> tuple:
    """Return (kept_box_iter, n_total, n_kept) after applying filter_module.

    Args:
        boxes_obj: Ultralytics boxes object for one image.
        img: Grayscale image array the boxes were predicted on.
        filter_cfg: Filter configuration passed to filter_module.
        filter_module: Module exposing filter_ultralytics_results().
        total_rejects: Mapping of rejection reason -> count, updated in place.
    """
    keep_idx, rejects = filter_module.filter_ultralytics_results(img, boxes_obj, filter_cfg)
    for (_, reason) in rejects:
        total_rejects[reason] += 1
    return (boxes_obj[i] for i in keep_idx), len(boxes_obj), len(keep_idx)


def _print_filter_summary(n_kept: int, n_total: int, total_rejects: dict) -> None:
    """Print how many predictions were filtered out and why.

    Args:
        n_kept: Number of predictions kept after filtering.
        n_total: Total number of predictions before filtering.
        total_rejects: Mapping of rejection reason -> count.
    """
    if not total_rejects and n_total == n_kept:
        return
    pct = (1 - n_kept / max(n_total, 1)) * 100
    print(f"  Predictions: kept {n_kept}/{n_total} ({pct:.1f}% filtered)")
    for reason, cnt in sorted(total_rejects.items()):
        print(f"    rejected for {reason}: {cnt}")


def run_tiled_inference(model, pairs: list[tuple[Path, str]], min_thresh: float, filter_cfg=None,
                        filter_module=None, edge_pad_px: int = 0) -> tuple[dict, set]:
    """Run model on each tile, aggregate max-conf per (source_stem, class).

    Args:
        model: Loaded RTDETR (or YOLO) model.
        pairs: List of (tile_path, source_stem) pairs from collect_tiled_inputs().
        min_thresh: Minimum confidence threshold to pass to the model.
        filter_cfg: Optional filter configuration passed to filter_module.
        filter_module: Optional module exposing filter_ultralytics_results().
        edge_pad_px: Number of pixels to zero out around each tile's border before inference.
    """
    preds_max_conf = defaultdict(lambda: defaultdict(float))
    all_stems = set()
    total_rejects = defaultdict(int)
    n_total = n_kept = 0

    n = len(pairs)
    for i, (tile_path, src_stem) in enumerate(pairs):
        if i % 200 == 0:
            print(f"  {i}/{n} tiles processed...", flush=True)
        all_stems.add(src_stem)

        img = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if edge_pad_px > 0:
            img = _apply_edge_pad(img, edge_pad_px)
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        results = model.predict(source=img_3ch, conf=min_thresh, max_det=300,
                                verbose=False, save=False)
        if not results or len(results[0].boxes) == 0:
            continue

        boxes_obj = results[0].boxes.cpu()
        if filter_cfg is not None and filter_module is not None:
            box_iter, t, k = _apply_filters(boxes_obj, img, filter_cfg, filter_module, total_rejects)
        else:
            box_iter, t, k = boxes_obj, len(boxes_obj), len(boxes_obj)
        n_total += t
        n_kept += k
        _record_max_conf(preds_max_conf, src_stem, box_iter)

    print(f"  {n}/{n} tiles processed.")
    if filter_cfg is not None:
        _print_filter_summary(n_kept, n_total, total_rejects)
    return dict(preds_max_conf), all_stems


# Full-res mode
def collect_fullres_inputs(images_dir: Path, split_list: list[str]) -> list[tuple[Path, str]]:
    """Pair each split stem with its image path; warn if any are missing.

    Args:
        images_dir: Flat directory of full-res PNGs.
        split_list: Stems to look up, as returned by load_split_list().
    """
    pairs = []
    missing = 0
    for stem in split_list:
        img_path = images_dir / f"{stem}.png"
        if img_path.exists():
            pairs.append((img_path, stem))
        else:
            missing += 1
    if missing:
        print(f"  WARNING: {missing} stems in split list have no corresponding PNG in {images_dir}")
    return pairs


def aggregate_gt_fullres(pairs: list[tuple[Path, str]], labels_dir: Path) -> dict[str, set]:
    """Collect GT class IDs per source image from its label file.

    Args:
        pairs: List of (image_path, stem) pairs from collect_fullres_inputs().
        labels_dir: Flat directory of YOLO .txt label files.
    """
    gt_per_image = defaultdict(set)
    for img_path, stem in pairs:
        for c in load_yolo_class_ids(labels_dir / f"{stem}.txt"):
            gt_per_image[stem].add(c)
        if stem not in gt_per_image:
            gt_per_image[stem] = set()
    return dict(gt_per_image)


def run_fullres_inference(model, pairs: list[tuple[Path, str]], min_thresh: float, imgsz: int,
                          filter_cfg=None, filter_module=None, edge_pad_px: int = 0) -> tuple[dict, set]:
    """Run model on each full-res image at the given imgsz.

    Args:
        model: Loaded RTDETR (or YOLO) model.
        pairs: List of (image_path, stem) pairs from collect_fullres_inputs().
        min_thresh: Minimum confidence threshold to pass to the model.
        imgsz: Inference image size; larger catches small markers better but is slower.
        filter_cfg: Optional filter configuration passed to filter_module.
        filter_module: Optional module exposing filter_ultralytics_results().
        edge_pad_px: Number of pixels to zero out around each image's border before inference.
    """
    preds_max_conf = defaultdict(lambda: defaultdict(float))
    all_stems = set()
    total_rejects = defaultdict(int)
    n_total = n_kept = 0

    n = len(pairs)
    for i, (img_path, stem) in enumerate(pairs):
        if i % 100 == 0:
            print(f"  {i}/{n} images processed...", flush=True)
        all_stems.add(stem)

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if edge_pad_px > 0:
            img = _apply_edge_pad(img, edge_pad_px)
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        results = model.predict(source=img_3ch, conf=min_thresh, max_det=300,
                                imgsz=imgsz, verbose=False, save=False)
        if not results or len(results[0].boxes) == 0:
            continue

        boxes_obj = results[0].boxes.cpu()
        if filter_cfg is not None and filter_module is not None:
            box_iter, t, k = _apply_filters(boxes_obj, img, filter_cfg, filter_module, total_rejects)
        else:
            box_iter, t, k = boxes_obj, len(boxes_obj), len(boxes_obj)
        n_total += t
        n_kept += k
        _record_max_conf(preds_max_conf, stem, box_iter)

    print(f"  {n}/{n} images processed.")
    if filter_cfg is not None:
        _print_filter_summary(n_kept, n_total, total_rejects)
    return dict(preds_max_conf), all_stems


# Metrics

def compute_image_level_metrics(gt_per_image: dict, preds_max_conf: dict, all_stems: set,
                                 class_id: int, threshold: float) -> dict:
    """Tally TP/FP/FN/TN over all images for one class at one threshold.

    Args:
        gt_per_image: Mapping of stem -> set of ground-truth class IDs.
        preds_max_conf: Mapping of stem -> class_id -> max predicted confidence.
        all_stems: Set of all image stems to evaluate over.
        class_id: Class ID to compute metrics for.
        threshold: Confidence threshold above which a prediction counts as positive.
    """
    tp = fn = fp = tn = 0
    for stem in all_stems:
        gt_has = class_id in gt_per_image.get(stem, set())
        pred_has = preds_max_conf.get(stem, {}).get(class_id, 0.0) >= threshold

        if gt_has and pred_has:
            tp += 1
        elif gt_has:
            fn += 1
        elif pred_has:
            fp += 1
        else:
            tn += 1

    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0)
        else float("nan")
    )
    accuracy = (tp + tn) / total if total > 0 else float("nan")

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
        "threshold": threshold,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_images": total,
        "n_positive": tp + fn,
        "n_negative": fp + tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Detection-level mAP / AP50 / AP75
# ---------------------------------------------------------------------------

def _yolo_boxes_to_xyxy(label_path: Path, img_w: int, img_h: int):
    """Read a YOLO label file and return (class_ids, boxes_xyxy) as arrays.

    YOLO format: class cx cy w h  (all normalised to [0,1]).
    Returns parallel lists so callers can zip them.
    """
    class_ids, boxes = [], []
    if not label_path.exists():
        return class_ids, boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            class_ids.append(c)
            boxes.append([x1, y1, x2, y2])
    return class_ids, boxes


def _iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between every GT/pred pair.  Returns shape (n_gt, n_pred)."""
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    # Intersection
    inter_x1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    inter_y1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    inter_x2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    inter_y2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    gt_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    )[:, None]
    pred_area = (
        (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    )[None, :]

    union_area = gt_area + pred_area - inter_area
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated AP (COCO-style)."""
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 101):
        p = precisions[recalls >= thr]
        ap += p.max() if p.size > 0 else 0.0
    return ap / 101.0


def _compute_ap_for_class(
    gt_per_image: dict,          # stem -> set of class_ids  (used only for presence)
    gt_boxes_per_image: dict,    # stem -> list of [x1,y1,x2,y2] for this class
    pred_records: list,          # list of (stem, conf, [x1,y1,x2,y2])
    iou_threshold: float,
    n_gt_total: int,
) -> float:
    """Standard VOC/COCO AP for a single class at a single IoU threshold."""
    if n_gt_total == 0:
        return float("nan")

    # Sort predictions by descending confidence
    pred_records_sorted = sorted(pred_records, key=lambda x: -x[1])

    tp_arr = np.zeros(len(pred_records_sorted), dtype=np.float32)
    fp_arr = np.zeros(len(pred_records_sorted), dtype=np.float32)

    # Track which GT boxes have already been matched (per image)
    matched_gt: dict[str, set] = defaultdict(set)

    for idx, (stem, conf, pred_box) in enumerate(pred_records_sorted):
        gt_boxes = gt_boxes_per_image.get(stem, [])
        if not gt_boxes:
            fp_arr[idx] = 1
            continue

        gt_arr = np.array(gt_boxes, dtype=np.float32)
        pred_arr = np.array([pred_box], dtype=np.float32)
        ious = _iou_matrix(gt_arr, pred_arr)[:, 0]  # shape (n_gt,)

        best_gt = int(np.argmax(ious))
        if ious[best_gt] >= iou_threshold and best_gt not in matched_gt[stem]:
            tp_arr[idx] = 1
            matched_gt[stem].add(best_gt)
        else:
            fp_arr[idx] = 1

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)

    recalls = tp_cum / n_gt_total
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    return _ap_from_pr(recalls, precisions)


def compute_detection_ap(
    model,
    pairs: list,
    labels_dir: Path,
    min_conf: float,
    iou_thresholds: tuple = (0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    imgsz: int | None = None,
    mode: str = "tiled",
) -> dict:
    """Run inference and compute AP50, AP75, and mAP@[0.50:0.95] per class.

    Parameters
    ----------
    model       : loaded RTDETR (or YOLO) model
    pairs       : list of (image_path, stem) as built by collect_*_inputs()
    labels_dir  : directory containing flat YOLO .txt label files
    min_conf    : minimum confidence to pass to the model (use lowest desired)
    iou_thresholds : IoU thresholds to sweep for mAP; default = COCO standard
    imgsz       : inference size (fullres mode); None = model default (tiled)
    mode        : "tiled" or "fullres" (controls stem extraction for GT lookup)

    Returns
    -------
    dict keyed by class_id, each value a dict with keys:
        ap50, ap75, map_50_95, n_gt
    Also has key "all" with macro-averaged values across classes.
    """
    # ------------------------------------------------------------------
    # 1. Collect GT boxes per class per image
    # ------------------------------------------------------------------
    # gt_boxes[cls_id][stem] = list of [x1,y1,x2,y2]
    gt_boxes_per_class: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n_gt_per_class: dict[int, int] = defaultdict(int)

    for img_path, stem in pairs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = (img.shape[:2] if img is not None else (1, 1))

        lbl_stem = img_path.stem if mode == "tiled" else stem
        lbl_path = labels_dir / f"{lbl_stem}.txt"
        cls_ids, boxes = _yolo_boxes_to_xyxy(lbl_path, w, h)

        # For tiled mode the GT stem is the tile stem; aggregate to src later
        agg_stem = stem_from_tile(img_path) if mode == "tiled" else stem
        for c, box in zip(cls_ids, boxes):
            gt_boxes_per_class[c][agg_stem].append(box)
            n_gt_per_class[c] += 1

    # ------------------------------------------------------------------
    # 2. Run inference; collect (stem, conf, xyxy) per class
    # ------------------------------------------------------------------
    # pred_records[cls_id] = [(stem, conf, [x1,y1,x2,y2]), ...]
    pred_records: dict[int, list] = defaultdict(list)

    n = len(pairs)
    predict_kwargs = dict(conf=min_conf, max_det=300, verbose=False, save=False)
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    for i, (img_path, stem) in enumerate(pairs):
        if i % 200 == 0:
            print(f"  [AP] {i}/{n} ...", flush=True)

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

        results = model.predict(source=img_3ch, **predict_kwargs)
        if not results or len(results[0].boxes) == 0:
            continue

        agg_stem = stem_from_tile(img_path) if mode == "tiled" else stem
        boxes_obj = results[0].boxes.cpu()
        xyxy_all = boxes_obj.xyxy.numpy()   # (N,4) absolute pixel coords
        for j, box in enumerate(boxes_obj):
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = xyxy_all[j].tolist()
            pred_records[cls_id].append((agg_stem, conf, [x1, y1, x2, y2]))

    print(f"  [AP] {n}/{n} done.", flush=True)

    # ------------------------------------------------------------------
    # 3. Compute AP at each IoU threshold, then average
    # ------------------------------------------------------------------
    results_out: dict = {}
    all_ap25, all_ap50, all_ap75, all_map = [], [], [], []

    for cls_id in sorted(CLASS_NAMES):
        n_gt = n_gt_per_class.get(cls_id, 0)
        preds = pred_records.get(cls_id, [])
        gt_boxes = gt_boxes_per_class.get(cls_id, {})

        aps = {}
        for iou_thr in iou_thresholds:
            aps[iou_thr] = _compute_ap_for_class(
                gt_per_image={},   # not used internally
                gt_boxes_per_image=gt_boxes,
                pred_records=preds,
                iou_threshold=iou_thr,
                n_gt_total=n_gt,
            )

        ap25   = aps.get(0.25, float("nan"))
        ap50   = aps.get(0.50, float("nan"))
        ap75   = aps.get(0.75, float("nan"))
        # mAP@[0.50:0.95] excludes the ap25 point to match COCO convention
        valid_coco = [v for thr, v in aps.items() if thr >= 0.50 and not np.isnan(v)]
        map_50_95 = float(np.mean(valid_coco)) if valid_coco else float("nan")

        results_out[cls_id] = {
            "class_name": CLASS_NAMES[cls_id],
            "n_gt": n_gt,
            "ap25": ap25,
            "ap50": ap50,
            "ap75": ap75,
            "map_50_95": map_50_95,
            "ap_per_iou": aps,
        }

        if not np.isnan(ap25):    all_ap25.append(ap25)
        if not np.isnan(ap50):    all_ap50.append(ap50)
        if not np.isnan(ap75):    all_ap75.append(ap75)
        if not np.isnan(map_50_95): all_map.append(map_50_95)

    results_out["all"] = {
        "class_name": "ALL (macro)",
        "n_gt": sum(n_gt_per_class.values()),
        "ap25":      float(np.mean(all_ap25))   if all_ap25 else float("nan"),
        "ap50":      float(np.mean(all_ap50))   if all_ap50 else float("nan"),
        "ap75":      float(np.mean(all_ap75))   if all_ap75 else float("nan"),
        "map_50_95": float(np.mean(all_map))    if all_map  else float("nan"),
        "ap_per_iou": {},
    }

    return results_out


def print_ap_table(ap_results: dict) -> None:
    """Pretty-print the dict returned by compute_detection_ap."""
    print()
    print("=" * 72)
    print("  DETECTION AP  (box-level, COCO IoU sweep)")
    print("=" * 72)
    print(f"  {'Class':<12} {'n_GT':>6}  {'AP@25':>7}  {'AP@50':>7}  {'AP@75':>7}  {'mAP@.5:.95':>10}")
    print("-" * 72)
    for key in list(sorted(k for k in ap_results if k != "all")) + ["all"]:
        r = ap_results[key]
        def _p(v):
            return f"{v*100:7.2f}%" if not np.isnan(v) else "    --  "
        print(f"  {r['class_name']:<12} {r['n_gt']:>6}  {_p(r['ap25'])}  {_p(r['ap50'])}  "
              f"{_p(r['ap75'])}  {_p(r['map_50_95']):>10}")
    print("=" * 72)


def write_ap_csv(ap_results: dict, csv_path: Path) -> None:
    """Write AP results to a CSV file."""
    fieldnames = ["class_id", "class_name", "n_gt", "ap25", "ap50", "ap75", "map_50_95"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, r in ap_results.items():
            writer.writerow({
                "class_id":   key,
                "class_name": r["class_name"],
                "n_gt":       r["n_gt"],
                "ap25":       r["ap25"],
                "ap50":       r["ap50"],
                "ap75":       r["ap75"],
                "map_50_95":  r["map_50_95"],
            })
    print(f"Wrote: {csv_path}")

def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  -- "
    return f"{v*100:5.1f}%"


def fmt_count(v, width=4):
    return f"{v:>{width}d}"


def print_table(rows_per_class):
    for cls_id in sorted(rows_per_class):
        rows = rows_per_class[cls_id]
        if not rows:
            continue
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        n_pos = rows[0]["n_positive"]
        n_neg = rows[0]["n_negative"]
        n_tot = rows[0]["n_images"]
        print()
        print("=" * 90)
        print(f"  {cls_name.upper()} (class {cls_id})")
        print(f"    Images with this class:    {n_pos}")
        print(f"    Images WITHOUT this class: {n_neg}")
        print(f"    Total images:              {n_tot}")
        print("=" * 90)
        print(f"  {'conf':>5} | {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} | "
              f"{'Prec':>6} {'Rec':>6} {'Spec':>6} {'F1':>6} {'Acc':>6}")
        print("-" * 75)
        for r in rows:
            print(
                f"  {r['threshold']:>5.2f} | "
                f"{fmt_count(r['tp'])} {fmt_count(r['fn'])} "
                f"{fmt_count(r['fp'])} {fmt_count(r['tn'])} | "
                f"{fmt_pct(r['precision'])} {fmt_pct(r['recall'])} "
                f"{fmt_pct(r['specificity'])} {fmt_pct(r['f1'])} "
                f"{fmt_pct(r['accuracy'])}"
            )


def write_csv(rows_per_class, csv_path):
    fieldnames = [
        "class_id", "class_name", "threshold",
        "tp", "fn", "fp", "tn",
        "n_images", "n_positive", "n_negative",
        "precision", "recall", "specificity", "f1", "accuracy",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls_id in sorted(rows_per_class):
            for r in rows_per_class[cls_id]:
                writer.writerow(r)
    print(f"\nWrote: {csv_path}")


def _build_parser():
    p = argparse.ArgumentParser(description="Image-level (per source image) classification metrics")
    p.add_argument("--mode", choices=["tiled", "fullres"], default="tiled",
                   help="tiled = aggregate tiles (val/test); fullres = run on full-res images")
    p.add_argument("--weights", required=True, type=str)

    # Tiled mode
    p.add_argument("--data-yaml", type=str, default=None, help="(tiled) data.yaml path")
    p.add_argument("--split", default="val", choices=["val", "test"], help="(tiled) split subdir to read")

    # Fullres mode
    p.add_argument("--images-dir", type=str, default=None, help="(fullres) flat dir of full-res PNGs")
    p.add_argument("--labels-dir", type=str, default=None, help="(fullres) flat dir of YOLO .txt labels")
    p.add_argument("--split-list", type=str, default=None, help="(fullres) e.g. splits/train.txt")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="(fullres) inference imgsz; larger catches small markers better but slower")

    p.add_argument("--output-dir", default="image_level_metrics")
    p.add_argument("--conf", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5],
                   help="Confidence threshold(s)")
    p.add_argument("--compute-ap", action="store_true",
                   help="Also compute box-level AP@25, AP@50, AP@75, and mAP@[0.50:0.95] "
                        "per class. Runs a second inference pass over the tiles/images.")
    p.add_argument("--ap-min-conf", type=float, default=0.05,
                   help="Min confidence for the AP inference pass (default 0.05).")
    p.add_argument("--ultralytics-fork-path", type=str,
                   default="ultralytics_mammo",
                   help="Path to the custom ultralytics fork.")

    # Filters (require inference_filters.py)
    p.add_argument("--apply-filters", action="store_true",
                   help="Apply inference-time filters (padding + edge by default)")
    p.add_argument("--filter-padding", action="store_true", default=True)
    p.add_argument("--no-filter-padding", dest="filter_padding", action="store_false")
    p.add_argument("--filter-edge", action="store_true", default=True)
    p.add_argument("--no-filter-edge", dest="filter_edge", action="store_false")
    p.add_argument("--filter-size", action="store_true", default=False,
                   help="Reject too-large predictions (default OFF)")
    p.add_argument("--padding-pixel-threshold", type=int, default=10)
    p.add_argument("--padding-area-threshold", type=float, default=0.20)
    p.add_argument("--edge-margin-px", type=int, default=15)
    p.add_argument("--max-box-area-pct", type=float, default=0.30)
    p.add_argument("--edge-pad-px", type=int, default=0,
                   help="Mask outer N pixels to black before inference. Try 10. Default 0 = off.")
    return p


def _resolve_inputs(args):
    """Validate mode-specific args and build (pairs, gt_per_image, mode_str)."""
    if args.mode == "tiled":
        if not args.data_yaml:
            print("ERROR: --data-yaml required in tiled mode")
            sys.exit(1)
        yaml_dir = Path(args.data_yaml).resolve().parent
        images_dir = yaml_dir / "images" / args.split
        labels_dir = yaml_dir / "labels" / args.split
        if not images_dir.exists() or not labels_dir.exists():
            print(f"ERROR: missing images/labels dir under {yaml_dir}")
            sys.exit(1)
        pairs = collect_tiled_inputs(images_dir)
        gt = aggregate_gt_tiled(pairs, labels_dir)
        mode_str = f"TILED  split={args.split}  {len(pairs)} tiles -> {len(gt)} source images"
        return pairs, gt, mode_str

    for arg_name, val in [("--images-dir", args.images_dir),
                          ("--labels-dir", args.labels_dir),
                          ("--split-list", args.split_list)]:
        if not val:
            print(f"ERROR: {arg_name} required in fullres mode")
            sys.exit(1)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    split_list_path = Path(args.split_list)
    for path, label in [(images_dir, "images-dir"), (labels_dir, "labels-dir"),
                        (split_list_path, "split-list")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)
    pairs = collect_fullres_inputs(images_dir, labels_dir, load_split_list(split_list_path))
    gt = aggregate_gt_fullres(pairs, labels_dir)
    return pairs, gt, f"FULLRES  imgsz={args.imgsz}  {len(pairs)} images"


def _build_filter_cfg(args):
    """Import and configure the optional inference_filters module."""
    if not args.apply_filters:
        return None, None
    try:
        import inference_filters as filter_module
    except ImportError as e:
        print(f"ERROR: --apply-filters requires inference_filters.py on PYTHONPATH. Error: {e}")
        sys.exit(1)
    cfg = filter_module.FilterConfig(
        apply_padding_filter=args.filter_padding,
        padding_pixel_threshold=args.padding_pixel_threshold,
        padding_area_threshold=args.padding_area_threshold,
        apply_edge_filter=args.filter_edge,
        edge_margin_px=args.edge_margin_px,
        apply_size_filter=args.filter_size,
        max_box_area_pct=args.max_box_area_pct,
    )
    return cfg, filter_module


def main():
    args = _build_parser().parse_args()

    if args.ultralytics_fork_path:
        sys.path.insert(0, args.ultralytics_fork_path)
    from ultralytics import RTDETR

    filter_cfg, filter_module = _build_filter_cfg(args)
    pairs, gt_per_image, mode_str = _resolve_inputs(args)

    if not pairs:
        print("ERROR: no images to process")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_unique = len(gt_per_image)

    print("=" * 70)
    print("Image-Level Classification Metrics")
    print("=" * 70)
    print(f"  Mode:           {mode_str}")
    print(f"  Weights:        {args.weights}")
    print(f"  Conf thresholds:{args.conf}")
    print(f"  Output dir:     {output_dir}")
    if filter_cfg:
        print("  Filters:        ON")
        print(f"    padding:      {'on' if filter_cfg.apply_padding_filter else 'off'} "
              f"(pixel<{filter_cfg.padding_pixel_threshold}, "
              f">{filter_cfg.padding_area_threshold:.0%} area)")
        print(f"    edge:         {'on' if filter_cfg.apply_edge_filter else 'off'} "
              f"(margin {filter_cfg.edge_margin_px}px)")
        print(f"    size:         {'on' if filter_cfg.apply_size_filter else 'off'} "
              f"(>{filter_cfg.max_box_area_pct:.0%} area)")
    else:
        print("  Filters:        OFF (use --apply-filters to enable)")
    print(f"  Edge pad:       {args.edge_pad_px}px" if args.edge_pad_px > 0 else "  Edge pad:       OFF")
    print("=" * 70, flush=True)

    print(f"\n  Source image counts ({n_unique} unique):")
    for cls_id, name in CLASS_NAMES.items():
        n_pos = sum(1 for s in gt_per_image.values() if cls_id in s)
        print(f"    {name:5s}: {n_pos} contain it, {n_unique - n_pos} do not")

    print("\nLoading model...", flush=True)
    model = RTDETR(args.weights)

    print("\nRunning inference...", flush=True)
    min_thresh = min(args.conf)
    runner = run_tiled_inference if args.mode == "tiled" else run_fullres_inference
    runner_kwargs = {"imgsz": args.imgsz} if args.mode == "fullres" else {}
    preds_max_conf, all_stems = runner(
        model, pairs, min_thresh,
        filter_cfg=filter_cfg, filter_module=filter_module,
        edge_pad_px=args.edge_pad_px, **runner_kwargs,
    )

    all_stems = all_stems | set(gt_per_image.keys())

    print("\nComputing metrics...", flush=True)
    rows_per_class = defaultdict(list)
    for cls_id in sorted(CLASS_NAMES):
        for thr in sorted(args.conf):
            rows_per_class[cls_id].append(
                compute_image_level_metrics(gt_per_image, preds_max_conf, all_stems, cls_id, thr)
            )

    print_table(rows_per_class)

    txt_path = output_dir / "image_level_metrics.txt"
    with open(txt_path, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print_table(rows_per_class)
        sys.stdout = original_stdout
    print(f"\nWrote: {txt_path}")
    write_csv(rows_per_class, output_dir / "image_level_metrics.csv")

    # ------------------------------------------------------------------
    # Optional box-level AP computation
    # ------------------------------------------------------------------
    if args.compute_ap:
        print("\nComputing detection AP (AP@25, AP@50, AP@75, mAP@[0.50:0.95])...", flush=True)

        # Resolve the labels directory for whichever mode we're in
        if args.mode == "tiled":
            yaml_dir = Path(args.data_yaml).resolve().parent
            ap_labels_dir = yaml_dir / "labels" / args.split
        else:
            ap_labels_dir = Path(args.labels_dir)

        ap_results = compute_detection_ap(
            model=model,
            pairs=pairs,
            labels_dir=ap_labels_dir,
            min_conf=args.ap_min_conf,
            mode=args.mode,
            imgsz=args.imgsz if args.mode == "fullres" else None,
        )

        print_ap_table(ap_results)

        ap_txt_path = output_dir / "detection_ap_metrics.txt"
        with open(ap_txt_path, "w") as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print_ap_table(ap_results)
            sys.stdout = original_stdout
        print(f"\nWrote: {ap_txt_path}")
        write_ap_csv(ap_results, output_dir / "detection_ap_metrics.csv")

    print()
    print("=" * 90)
    print("READING THE TABLE")
    print("=" * 90)
    print("  Recall      = of images with this marker, what % did the model find?")
    print("                (sensitivity, high = rarely misses)")
    print("  Specificity = of images WITHOUT this marker, what % were correctly")
    print("                NOT flagged? (high = few false alarms)")
    print("  Precision   = of images flagged for this marker, what % actually have it?")
    print("                (high = when model says yes, it's usually right)")
    print()
    print("Note on training-set metrics:")
    print("  Train recall is expected to be very high (95%+) since the model has")
    print("  seen this data. Low train recall = undertrained. Large train-vs-val gap = overfit.")


if __name__ == "__main__":
    main()