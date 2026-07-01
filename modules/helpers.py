import pydicom
import numpy as np 
import cv2
from pathlib import Path
from PIL import Image

def write_tile_yolo_labels(label_path, img_w, img_h, row_start, col_start,
                            tile_size, out_label_path, min_visibility=0.1):
    """Read a whole-image YOLO label file and write a tile-local YOLO label file.

    label_path:    path to the whole-image YOLO txt (class x_c y_c w h, all
                    normalized 0-1 relative to the full image). If None or the
                    file doesn't exist, an empty label file is written.
    img_w, img_h:  full image dimensions in pixels.
    row_start,
    col_start:     top-left pixel coords of this tile within the full image
                    (same values used to slice the tile in png_to_tiles /
                    dicom_to_tiles).
    tile_size:     tile side length in pixels.
    out_label_path: where to write the tile-local label txt.
    min_visibility: a box that gets clipped at the tile edge is kept only if
                    at least this fraction of its original area survives —
                    filters out slivers of boxes barely poking into the tile.

    Always writes a file (empty if there are no boxes in this tile), so every
    tile image has a matching label file, including negatives.
    """
    tile_lines = []

    if label_path is not None and Path(label_path).exists():
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        tx1, ty1 = col_start, row_start
        tx2, ty2 = col_start + tile_size, row_start + tile_size

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            cls = int(cls)
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)

            # normalized full-image box -> absolute pixel coords
            box_xc, box_yc = xc * img_w, yc * img_h
            box_w,  box_h  = w  * img_w, h  * img_h
            x1, y1 = box_xc - box_w / 2, box_yc - box_h / 2
            x2, y2 = box_xc + box_w / 2, box_yc + box_h / 2

            # intersect box with tile bounds
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            iw, ih = ix2 - ix1, iy2 - iy1
            if iw <= 0 or ih <= 0:
                continue  # box doesn't overlap this tile

            orig_area  = box_w * box_h
            inter_area = iw * ih
            if orig_area > 0 and (inter_area / orig_area) < min_visibility:
                continue  # too little of the box survives clipping

            # clipped box, re-normalized relative to the tile
            new_xc = ((ix1 + ix2) / 2 - tx1) / tile_size
            new_yc = ((iy1 + iy2) / 2 - ty1) / tile_size
            new_w  = iw / tile_size
            new_h  = ih / tile_size

            tile_lines.append(f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

    out_label_path = Path(out_label_path)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        f.write("\n".join(tile_lines))
        if tile_lines:
            f.write("\n")

    return len(tile_lines)


def png_to_tiles(data_root, png_rel_path, tile_size=640, drop_black_tiles=False,
                  label_path=None, labels_out_dir=None, min_visibility=0.1):
    """Load a PNG and slice into non-overlapping tiles.

    Returns a list of (tile_bgr_array, src_stem) pairs in the same format
    that run_tiled_inference expects — no changes to that function needed.
    Partial tiles at the right/bottom edge are zero-padded to tile_size.

    If drop_black_tiles is True, tiles that are completely black
    (all pixel values == 0) are skipped and not included in the output.

    If label_path and labels_out_dir are given, whole-image YOLO labels at
    label_path are converted to tile-local coordinates and written to
    labels_out_dir as "{stem}__{tile_idx:02d}.txt", one per kept tile — using
    the same tile_idx that the disk-writing code assigns via enumerate(tiles),
    so image and label filenames stay in sync even when tiles are dropped.
    """
    stem     = Path(png_rel_path).name[:-4]
    png_path = Path(data_root) / png_rel_path
    pil_img  = Image.open(png_path).convert("L")  # 8-bit grayscale
    arr      = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)
    H, W     = arr.shape[:2]

    tiles = []
    tile_idx = 0
    for row_start in range(0, H, tile_size):
        for col_start in range(0, W, tile_size):
            patch = arr[row_start:row_start + tile_size,
                        col_start:col_start + tile_size]
            ph, pw = patch.shape[:2]
            if ph < tile_size or pw < tile_size:   # zero-pad partial edge tiles
                pad = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad[:ph, :pw] = patch
                patch = pad

            if drop_black_tiles and not np.any(patch):
                continue

            if labels_out_dir is not None:
                out_label_path = Path(labels_out_dir) / f"{stem}__{tile_idx:02d}.txt"
                write_tile_yolo_labels(
                    label_path, W, H, row_start, col_start, tile_size,
                    out_label_path, min_visibility=min_visibility,
                )

            tiles.append((patch, stem))
            tile_idx += 1
    return tiles


def dicom_to_tiles(data_root, dcm_rel_path, tile_size=640, drop_black_tiles=False,
                    label_path=None, labels_out_dir=None, min_visibility=0.1):
    """Convert a DICOM to 8-bit BGR and slice into non-overlapping tiles.

    Returns a list of (tile_bgr_array, src_stem) pairs in the same format
    that run_tiled_inference expects — no changes to that function needed.
    Partial tiles at the right/bottom edge are zero-padded to tile_size.

    If drop_black_tiles is True, tiles that are completely black
    (all pixel values == 0) are skipped and not included in the output.

    If label_path and labels_out_dir are given, whole-image YOLO labels at
    label_path are converted to tile-local coordinates and written to
    labels_out_dir as "{stem}__{tile_idx:02d}.txt", one per kept tile — using
    the same tile_idx that the disk-writing code assigns via enumerate(tiles),
    so image and label filenames stay in sync even when tiles are dropped.
    """
    stem    = Path(dcm_rel_path).name
    pil_img = dicom_to_8bit_pil(data_root, dcm_rel_path)
    arr     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)
    H, W    = arr.shape[:2]

    tiles = []
    tile_idx = 0
    for row_start in range(0, H, tile_size):
        for col_start in range(0, W, tile_size):
            patch = arr[row_start:row_start + tile_size,
                        col_start:col_start + tile_size]
            ph, pw = patch.shape[:2]
            if ph < tile_size or pw < tile_size:   # zero-pad partial edge tiles
                pad = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad[:ph, :pw] = patch
                patch = pad

            if drop_black_tiles and not np.any(patch):
                continue

            if labels_out_dir is not None:
                out_label_path = Path(labels_out_dir) / f"{stem}__{tile_idx:02d}.txt"
                write_tile_yolo_labels(
                    label_path, W, H, row_start, col_start, tile_size,
                    out_label_path, min_visibility=min_visibility,
                )

            tiles.append((patch, stem))
            tile_idx += 1
    return tiles

def dicom_to_8bit_pil(SOURCE, dcm_path):
    dcm = pydicom.dcmread(f"{SOURCE}/{dcm_path}")
    arr = dcm.pixel_array.astype(np.float32)

    # apply rescale slope/intercept if present
    slope  = getattr(dcm, 'RescaleSlope',     1)
    intercept = getattr(dcm, 'RescaleIntercept', 0)
    arr = arr * slope + intercept

    # normalize to 8-bit
    arr -= arr.min()
    arr /= arr.max() if arr.max() != 0 else 1
    arr  = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr)

def flip_lat(laterality, fov_horizontal_flip):
    if fov_horizontal_flip == 'YES':
        if laterality == 'R':
            laterality = 'L'
        else:
            laterality = 'R'

    return laterality