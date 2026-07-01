import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import Union, Optional

try:
    from IPython.display import display
except ImportError:
    display = None


# ============================================================
# Basic image helpers
# ============================================================

def downsample_image(
    gray_img: Union[np.ndarray, Image.Image],
    target_width: int = 400,
) -> tuple[np.ndarray, float]:
    """
    Parameters
    ----------
    gray_img : np.ndarray or PIL.Image
        Input grayscale image. Color images are converted to grayscale
        automatically.
    target_width : int, optional
        Width in pixels to resize the image to. Default is 400.

    Returns
    -------
    downsampled : np.ndarray
        Resized grayscale image.
    scale : float
        Ratio of target_width to the original image width.
    """
    gray_img = np.asarray(gray_img)

    if gray_img.ndim == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

    H, W = gray_img.shape
    scale = target_width / W
    target_height = int(H * scale)

    downsampled = cv2.resize(
        gray_img,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    return downsampled, scale


def extract_top_brightest_pixels(
    gray_img: Union[np.ndarray, Image.Image],
    top_percent: float = 5,
) -> np.ndarray:
    """
    Parameters
    ----------
    gray_img : np.ndarray or PIL.Image
        Input grayscale image. Color images are converted to grayscale
        automatically.
    top_percent : float, optional
        Percentage of pixel values to keep (counted from the brightest end).
        For example, ``top_percent=5`` keeps the brightest 5 % of pixels.
        Default is 5.

    Returns
    -------
    result : np.ndarray
        Grayscale image of the same shape as the input where all pixels below
        the computed threshold are set to 0, and pixels at or above the
        threshold retain their original values.
    """
    gray_img = np.asarray(gray_img)

    if gray_img.ndim == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

    threshold = np.percentile(gray_img, 100 - top_percent)
    bright_mask = gray_img >= threshold

    result = np.zeros_like(gray_img)
    result[bright_mask] = gray_img[bright_mask]

    return result


def _get_search_band(W: int, laterality: Union[str, object]) -> tuple[int, int]:
    """
    Parameters
    ----------
    W : int
        Total width of the image in pixels.
    laterality : str
        ``'L'`` for left breast or ``'R'`` for right breast (case-insensitive).

    Returns
    -------
    x_start : int
        Left bound of the search band (inclusive).
    x_end : int
        Right bound of the search band (exclusive).

    Raises
    ------
    ValueError
        If ``laterality`` is not ``'L'`` or ``'R'``.
    """
    laterality = str(laterality).upper()
    three_quarters = int(W * 0.75)

    if laterality == "R":
        return 0, three_quarters
    elif laterality == "L":
        return W - three_quarters, W
    else:
        raise ValueError("laterality must be 'L' or 'R'")

# ============================================================
# Edge / segment helpers
# ============================================================

def get_bright_segments_at_column(
    roi: np.ndarray,
    x_col: int,
) -> list[tuple[int, int]]:
    """
    Parameters
    ----------
    roi : np.ndarray
        2-D grayscale image or sub-region. 
    x_col : int
        Column index within ``roi`` to scan.

    Returns
    -------
    segments : list of tuple[int, int]
        Each element is a ``(start_y, end_y)`` pair (both inclusive) that
        describes one contiguous bright run within the column. The list is
        ordered top-to-bottom and may be empty if no bright pixels exist.
    """
    column = roi[:, x_col]
    bright_mask = column > 0

    segments = []
    in_segment = False
    start_y = 0

    for y in range(len(bright_mask)):
        if bright_mask[y] and not in_segment:
            start_y = y
            in_segment = True
        elif not bright_mask[y] and in_segment:
            segments.append((start_y, y - 1))
            in_segment = False

    if in_segment:
        segments.append((start_y, len(bright_mask) - 1))

    return segments


def vertical_midline_checker(
    gray_img: np.ndarray,
    laterality: Union[str, object],
    threshold: int = 1,
    step_back_pixels: int = 50,
) -> tuple[int, int]:
    """
    Parameters
    ----------
    gray_img : np.ndarray
        2-D grayscale image (already downsampled / filtered).
    laterality : str
        ``'L'`` or ``'R'`` (case-insensitive).
    threshold : int, optional
        Pixel value above which a pixel is considered "bright". Default is 1.
    step_back_pixels : int, optional
        Number of columns to step back from the detected breast edge before
        counting bright pixels. Default is 50.

    Returns
    -------
    num_above : int
        Number of pixels in the check column whose value exceeds
        ``threshold``.
    check_col : int
        The column index that was used for the bright-pixel count.

    Raises
    ------
    ValueError
        If ``laterality`` is not ``'L'`` or ``'R'``.
    """
    gray_img = np.asarray(gray_img)
    H, W = gray_img.shape
    laterality = str(laterality).upper()

    if laterality == "R":
        start_col = 0
        end_col = W // 2
        direction = 1
    elif laterality == "L":
        start_col = W - 1
        end_col = W // 2
        direction = -1
    else:
        raise ValueError("laterality must be 'L' or 'R'")

    breast_edge_col = None
    min_tissue_pixels = H * 0.10

    col = start_col
    while (direction == 1 and col < end_col) or (direction == -1 and col > end_col):
        col_vals = gray_img[:, col]
        num_above = np.sum(col_vals > threshold)

        if num_above >= min_tissue_pixels:
            breast_edge_col = col
            break

        col += direction

    if breast_edge_col is None:
        check_col = W // 4 if laterality == "R" else 3 * W // 4
    else:
        check_col = breast_edge_col - (direction * step_back_pixels)
        check_col = max(0, min(W - 1, check_col))

    col_vals = gray_img[:, check_col]
    num_above = int(np.sum(col_vals > threshold))

    return num_above, check_col


def get_boundary_edges_prop(
    roi: np.ndarray,
    global_x_offset: int,
    laterality: Optional[Union[str, object]] = None,
    search_margin: int = 50,
    min_width_fraction: float = 0.5,
) -> tuple[int, int, bool]:
    """
    Parameters
    ----------
    roi : np.ndarray
        2-D grayscale sub-image to search. Pixels with value > 0 are
        considered bright.
    global_x_offset : int
        X offset of the ROI within the full image. Returned edge coordinates
        are expressed in the full-image coordinate system.
    laterality : str or None, optional
        Unused in this function but kept for API consistency.
        Default is ``None``.
    search_margin : int, optional
        Maximum number of columns to scan inward from each horizontal edge
        when looking for the primary boundary. Default is 50.
    min_width_fraction : float, optional
        Minimum fraction of the ROI width that must be spanned by bright
        pixels in at least one row for the edges to be considered valid.
        Default is 0.5.

    Returns
    -------
    left_edge_global : int
        Column index of the detected left edge in the full image.
        0 when no valid edge pair is found.
    right_edge_global : int
        Column index of the detected right edge in the full image.
        0 when no valid edge pair is found.
    found : bool
        ``True`` if both edges were located and all validation checks pass;
        ``False`` otherwise.
    """
    H, W = roi.shape
    mask = roi > 0
    min_pixels = 10

    def _max_row_span(mask: np.ndarray) -> int:
        """
        Parameters
        ----------
        mask : np.ndarray
            2-D boolean array where ``True`` marks bright pixels.

        Returns
        -------
        int
            The widest single-row span, or 0 if no row has at least two
            bright pixels.
        """
        spans = []
        for row in mask:
            bright = np.where(row)[0]
            if bright.size >= 2:
                spans.append(int(bright[-1] - bright[0] + 1))
        return max(spans) if spans else 0

    left_found = False
    left_edge = 0
    for x in range(min(search_margin, W)):
        if np.any(mask[:, x]):
            left_edge = x
            left_found = True
            break

    right_found = False
    right_edge = W - 1
    for x in range(W - 1, max(W - search_margin - 1, -1), -1):
        if np.any(mask[:, x]):
            right_edge = x
            right_found = True
            break

    if left_found and right_found:
        left_bright_count = np.sum(mask[:, left_edge])
        right_bright_count = np.sum(mask[:, right_edge])
        if left_bright_count >= min_pixels and right_bright_count >= min_pixels:
            # Extra check: a real paddle must span min_width_fraction in a single row.
            if _max_row_span(mask) >= W * min_width_fraction:
                return left_edge + global_x_offset, right_edge + global_x_offset, True

    col_counts = mask.sum(axis=0)
    dense_cols = np.where(col_counts >= min_pixels)[0]

    if dense_cols.size == 0:
        return 0, 0, False

    left_edge = int(dense_cols.min())
    right_edge = int(dense_cols.max())
    span = right_edge - left_edge + 1

    if span < W * min_width_fraction:
        return 0, 0, False

    if _max_row_span(mask) < W * min_width_fraction:
        return 0, 0, False

    left_bright_count = np.sum(mask[:, left_edge])
    right_bright_count = np.sum(mask[:, right_edge])
    if left_bright_count < min_pixels or right_bright_count < min_pixels:
        return 0, 0, False

    return left_edge + global_x_offset, right_edge + global_x_offset, True

# ============================================================
# Core compression paddle logic
# ============================================================

def _analyze_compression_paddle(
    gray_img: Union[np.ndarray, Image.Image],
    laterality: Union[str, object],
    downsample_width: int = 400,
    top_percent: float = 7,
    min_width_fraction: float = 0.5,
) -> dict:
    """
    Parameters
    ----------
    gray_img : np.ndarray or PIL.Image
        Input mammogram image. Color images are converted to grayscale.
    laterality : str
        ``'L'`` or ``'R'`` (case-insensitive).
    downsample_width : int, optional
        Width in pixels for internal downsampling. Lower values are faster
        but may lose fine detail. Default is 400.
    top_percent : float, optional
        Percentage of brightest pixels to retain before analysis.
        Default is 7.
    min_width_fraction : float, optional
        Minimum fraction of the search band that a detected bright span must
        cover to be considered a real paddle boundary. Default is 0.5.

    Returns
    -------
    info : dict
        A dictionary containing:

        - ``"is_paddle"`` (*bool*): Whether a paddle was detected.
        - ``"laterality"`` (*str*): Laterality used.
        - ``"top_percent"`` (*float*): Brightness filter threshold used.
        - ``"downsample_width"`` (*int*): Internal image width used.
        - ``"original_shape"`` (*tuple[int, int]*): Shape of the input image.
        - ``"downsampled_shape"`` (*tuple[int, int]*): Shape after downsampling.
        - ``"scale"`` (*float*): Scale factor from original to downsampled.
        - ``"search_band"`` (*tuple[int, int]*): x-range searched.
        - ``"regions"`` (*list[dict]*): Per-region diagnostic details.
        - ``"regions_with_edges"`` (*list[str]*): Regions where edges were found.
        - ``"decision_reason"`` (*str*): Human-readable explanation of the
          final decision.
        - ``"vertical_count"`` (*int or None*): Bright pixel count from the
          vertical midline check.
        - ``"vertical_col"`` (*int or None*): Column used for the vertical
          midline check.
    """
    gray_img = np.asarray(gray_img)

    if gray_img.ndim == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

    laterality = str(laterality).upper()

    bright_only = extract_top_brightest_pixels(gray_img, top_percent=top_percent)
    img_small, scale = downsample_image(bright_only, target_width=downsample_width)

    H, W = img_small.shape
    third_h = H // 3
    x0, x1 = _get_search_band(W, laterality)

    info = {
        "is_paddle": False,
        "laterality": laterality,
        "top_percent": top_percent,
        "downsample_width": downsample_width,
        "original_shape": gray_img.shape,
        "downsampled_shape": img_small.shape,
        "scale": scale,
        "search_band": (x0, x1),
        "regions": [],
        "regions_with_edges": [],
        "decision_reason": "No valid paddle pattern found.",
        "vertical_count": None,
        "vertical_col": None,
    }

    regions = [
        ("top", 0, third_h, img_small[0:third_h, x0:x1]),
        ("bottom", 2 * third_h, H, img_small[2 * third_h:H, x0:x1]),
    ]

    for region_name, y_start, y_end, roi in regions:
        region_info = {
            "region": region_name,
            "y_range": (y_start, y_end),
            "found_edges": False,
            "left_edge": None,
            "right_edge": None,
            "left_segments": [],
            "right_segments": [],
            "passed": False,
            "reason": "No edges found.",
        }

        r_left, r_right, r_found = get_boundary_edges_prop(
            roi,
            x0,
            laterality=laterality,
            search_margin=50,
            min_width_fraction=min_width_fraction
        )

        if not r_found:
            info["regions"].append(region_info)
            continue

        region_info["found_edges"] = True
        region_info["left_edge"] = int(r_left)
        region_info["right_edge"] = int(r_right)
        info["regions_with_edges"].append(region_name)

        left_local = r_left - x0
        right_local = r_right - x0

        if not (0 <= left_local < roi.shape[1] and 0 <= right_local < roi.shape[1]):
            region_info["reason"] = "Edge column out of bounds."
            info["regions"].append(region_info)
            continue

        left_segments = get_bright_segments_at_column(roi, left_local)
        right_segments = get_bright_segments_at_column(roi, right_local)

        region_info["left_segments"] = left_segments
        region_info["right_segments"] = right_segments

        if len(left_segments) == 0 and len(right_segments) == 0:
            region_info["reason"] = "No segments on either edge."
            info["regions"].append(region_info)
            continue

        roi_height = roi.shape[0]
        boundary_margin = 10
        crop_artifact_max_thickness = 3  

        def _is_boundary_artifact(segments, roi_height, boundary_margin, max_thickness):
            """
            A segment is a boundary *artifact* (not a real paddle line) only if it's
            both confined to the outer margin AND thin enough to be noise rather
            than a genuine paddle edge.
            """
            if len(segments) < 1:
                return False 
            for s, e in segments:
                thickness = e - s + 1
                touches_top = s < boundary_margin
                touches_bottom = e > roi_height - boundary_margin
                if (touches_top or touches_bottom) and thickness > max_thickness:
                    return False  # a thick segment touching the edge is real signal, not artifact
            # only artifact if every segment is thin AND touches an edge
            return all(
                (s < boundary_margin or e > roi_height - boundary_margin)
                for s, e in segments
            ) if segments else False

        left_only_boundary = _is_boundary_artifact(left_segments, roi_height, boundary_margin, crop_artifact_max_thickness)
        right_only_boundary = _is_boundary_artifact(right_segments, roi_height, boundary_margin, crop_artifact_max_thickness)

        region_info["left_only_boundary"] = bool(left_only_boundary)
        region_info["right_only_boundary"] = bool(right_only_boundary)

        if left_only_boundary and right_only_boundary:
            region_info["reason"] = "Both edge segments are only boundary artifacts."
            info["regions"].append(region_info)
            continue

        if len(left_segments) > 0 and len(right_segments) > 0:
            left_y_coords = [y for seg in left_segments for y in range(seg[0], seg[1] + 1)]
            right_y_coords = [y for seg in right_segments for y in range(seg[0], seg[1] + 1)]

            left_center_y = float(np.mean(left_y_coords))
            right_center_y = float(np.mean(right_y_coords))
            center_diff = abs(left_center_y - right_center_y)
            max_center_diff = roi_height * 0.65

            region_info["left_center_y"] = left_center_y
            region_info["right_center_y"] = right_center_y
            region_info["center_diff"] = float(center_diff)
            region_info["max_center_diff"] = float(max_center_diff)

            if center_diff > max_center_diff:
                region_info["reason"] = "Left/right segment centers are too far apart."
                info["regions"].append(region_info)
                continue

            left_total_span = sum(end - start + 1 for start, end in left_segments)
            right_total_span = sum(end - start + 1 for start, end in right_segments)
            min_span = min(left_total_span, right_total_span)

            region_info["left_total_span"] = int(left_total_span)
            region_info["right_total_span"] = int(right_total_span)
        else:
            if len(left_segments) > 0:
                min_span = sum(end - start + 1 for start, end in left_segments)
                region_info["left_total_span"] = int(min_span)
                region_info["right_total_span"] = 0
            else:
                min_span = sum(end - start + 1 for start, end in right_segments)
                region_info["left_total_span"] = 0
                region_info["right_total_span"] = int(min_span)

        min_span_percent = 0.03
        min_required = roi_height * min_span_percent

        region_info["min_span"] = int(min_span)
        region_info["min_required"] = float(min_required)

        if min_span < min_required:
            region_info["reason"] = "Bright segment span is too small."
            info["regions"].append(region_info)
            continue

        vertical_count, col = vertical_midline_checker(img_small, laterality, threshold=1)
        info["vertical_count"] = int(vertical_count)
        info["vertical_col"] = int(col)

        if vertical_count < 12:
            region_info["reason"] = "Vertical check failed."
            info["regions"].append(region_info)
            continue

        region_info["passed"] = True
        region_info["reason"] = "Detected from valid edge/segment structure."
        info["regions"].append(region_info)

        info["is_paddle"] = True
        info["decision_reason"] = f"Detected from {region_name} region."
        return info

    if len(info["regions_with_edges"]) == 2:
        info["is_paddle"] = True
        info["decision_reason"] = "Detected because both top and bottom regions have edges."
        return info

    if len(info["regions_with_edges"]) > 0:
        info["decision_reason"] = (
            f"Only {len(info['regions_with_edges'])} region(s) had edges; "
            "need a stronger top/bottom paddle pattern."
        )

    return info


# ============================================================
# Annotation and debug display
# ============================================================

def annotate_compression_paddle(
    gray_img: Union[np.ndarray, Image.Image],
    laterality: Union[str, object],
    downsample_width: int = 400,
    top_percent: float = 1,
    min_width_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce annotated visualisations of the compression paddle search process.

    Draws the following overlays on both the original grayscale and the
    top-brightest-pixel version of the image:

    - **Yellow rectangles**: top-third and bottom-third search bands.
    - **Green vertical line**: column used for the vertical midline check.
    - **Blue lines**: segments and bounding box detected on the **left** edge.
    - **Red lines**: segments and bounding box detected on the **right** edge.
    - **Pink dashed line**: connection between the mean y-positions of the
      left and right edge segments.

    The annotations are produced at ``downsample_width`` resolution and then
    upsampled back to the original image dimensions before being returned.

    Parameters
    ----------
    gray_img : np.ndarray or PIL.Image
        Input mammogram image. Color images are converted to grayscale.
    laterality : str
        ``'L'`` or ``'R'`` (case-insensitive).
    downsample_width : int, optional
        Internal processing width in pixels. Default is 400.
    top_percent : float, optional
        Percentage of brightest pixels retained for the right-hand panel.
        Default is 1.
    min_width_fraction : float, optional
        Passed through to :func:`get_boundary_edges_prop`. Default is 0.5.

    Returns
    -------
    annotated_orig_full : np.ndarray
        RGB image (H × W × 3) of the original grayscale mammogram with
        annotation overlays, upsampled to the original resolution.
    annotated_bright_full : np.ndarray
        RGB image (H × W × 3) of the top-brightest-pixel image with
        annotation overlays, upsampled to the original resolution.
    """
    gray_img = np.asarray(gray_img)

    if gray_img.ndim == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

    laterality = str(laterality).upper()

    bright_only = extract_top_brightest_pixels(gray_img, top_percent=top_percent)
    img_small, scale = downsample_image(bright_only, target_width=downsample_width)
    img_small_orig, _ = downsample_image(gray_img, target_width=downsample_width)

    H, W = img_small.shape
    third_h = H // 3
    x0, x1 = _get_search_band(W, laterality)

    rgb_bright = Image.new("RGB", (W, H))
    rgb_bright.paste(Image.fromarray(img_small))
    draw_bright = ImageDraw.Draw(rgb_bright)

    rgb_orig = Image.new("RGB", (W, H))
    rgb_orig.paste(Image.fromarray(img_small_orig))
    draw_orig = ImageDraw.Draw(rgb_orig)

    rgb_bright.show()

    # Yellow top/bottom search boxes.
    draw_bright.rectangle([(x0, 0), (x1, third_h)], outline=(255, 255, 0), width=1)
    draw_bright.rectangle([(x0, 2 * third_h), (x1, H)], outline=(255, 255, 0), width=1)

    draw_orig.rectangle([(x0, 0), (x1, third_h)], outline=(255, 255, 0), width=1)
    draw_orig.rectangle([(x0, 2 * third_h), (x1, H)], outline=(255, 255, 0), width=1)

    rgb_bright.show()

    # Green vertical checker line.
    vertical_count, col = vertical_midline_checker(img_small, laterality, threshold=1)
    draw_bright.line([(col, 0), (col, H)], fill=(0, 255, 0), width=2)
    draw_orig.line([(col, 0), (col, H)], fill=(0, 255, 0), width=2)

    rgb_bright.show()

    # Process top and bottom regions.
    for region_y_start, region_y_end in [(0, third_h), (2 * third_h, H)]:
        region_roi = img_small[region_y_start:region_y_end, x0:x1]
        r_left, r_right, r_found = get_boundary_edges_prop(
            region_roi,
            x0,
            laterality=laterality,
            search_margin=50,
            min_width_fraction=min_width_fraction
        )

        if not r_found:
            continue

        left_local = r_left - x0
        right_local = r_right - x0

        if 0 <= left_local < region_roi.shape[1]:
            left_segments = get_bright_segments_at_column(region_roi, left_local)

            for y_start, y_end in left_segments:
                global_y_start = y_start + region_y_start
                global_y_end = y_end + region_y_start

                draw_bright.line([(r_left, global_y_start), (r_left, global_y_end)], fill=(0, 0, 255), width=3)
                draw_bright.line([(r_left, global_y_start), (r_right, global_y_start)], fill=(0, 0, 255), width=3)
                draw_bright.line([(r_left, global_y_end), (r_right, global_y_end)], fill=(0, 0, 255), width=3)

                draw_orig.line([(r_left, global_y_start), (r_left, global_y_end)], fill=(0, 0, 255), width=3)
                draw_orig.line([(r_left, global_y_start), (r_right, global_y_start)], fill=(0, 0, 255), width=3)
                draw_orig.line([(r_left, global_y_end), (r_right, global_y_end)], fill=(0, 0, 255), width=3)

            rgb_bright.show()
            if left_segments:
                all_left_y = [y for s, e in left_segments for y in range(s, e + 1)]
                left_mean_y = int(np.mean(all_left_y)) + region_y_start

        if 0 <= right_local < region_roi.shape[1]:
            right_segments = get_bright_segments_at_column(region_roi, right_local)

            for y_start, y_end in right_segments:
                global_y_start = y_start + region_y_start
                global_y_end = y_end + region_y_start

                draw_bright.line([(r_right, global_y_start), (r_right, global_y_end)], fill=(255, 0, 0), width=3)
                draw_bright.line([(r_right, global_y_start), (r_left, global_y_start)], fill=(255, 0, 0), width=3)
                draw_bright.line([(r_right, global_y_end), (r_left, global_y_end)], fill=(255, 0, 0), width=3)

                draw_orig.line([(r_right, global_y_start), (r_right, global_y_end)], fill=(255, 0, 0), width=3)
                draw_orig.line([(r_right, global_y_start), (r_left, global_y_start)], fill=(255, 0, 0), width=3)
                draw_orig.line([(r_right, global_y_end), (r_left, global_y_end)], fill=(255, 0, 0), width=3)

            rgb_bright.show()
            if right_segments:
                all_right_y = [y for s, e in right_segments for y in range(s, e + 1)]
                right_mean_y = int(np.mean(all_right_y)) + region_y_start

        # Draw pink dashed line between left and right edge mean y positions.
        if left_mean_y is not None and right_mean_y is not None:
            dash_length = 6
            gap_length = 4
            dx = r_right - r_left
            dy = right_mean_y - left_mean_y
            total = int(np.hypot(dx, dy))
            drawn = 0
            drawing = True

            while drawn < total:
                seg_end = min(drawn + (dash_length if drawing else gap_length), total)
                t0 = drawn / total
                t1 = seg_end / total

                if drawing:
                    x0_d = int(r_left + t0 * dx)
                    y0_d = int(left_mean_y + t0 * dy)
                    x1_d = int(r_left + t1 * dx)
                    y1_d = int(left_mean_y + t1 * dy)

                    draw_bright.line([(x0_d, y0_d), (x1_d, y1_d)], fill=(255, 105, 180), width=2)
                    draw_orig.line([(x0_d, y0_d), (x1_d, y1_d)], fill=(255, 105, 180), width=2)

                drawn = seg_end
                drawing = not drawing

            rgb_bright.show()

    annotated_bright = np.array(rgb_bright)
    annotated_orig = np.array(rgb_orig)

    H_orig, W_orig = gray_img.shape
    annotated_bright_full = cv2.resize(
        annotated_bright,
        (W_orig, H_orig),
        interpolation=cv2.INTER_NEAREST,
    )
    annotated_orig_full = cv2.resize(
        annotated_orig,
        (W_orig, H_orig),
        interpolation=cv2.INTER_NEAREST,
    )

    return annotated_orig_full, annotated_bright_full


def _show_compression_paddle_debug(
    gray_img: Union[np.ndarray, Image.Image],
    info: dict,
    img_name: Optional[str] = None,
) -> None:
    """
    Parameters
    ----------
    gray_img : np.ndarray or PIL.Image
        The original input mammogram image passed to the detector.
    info : dict
        The diagnostic dictionary returned by
        :func:`_analyze_compression_paddle`.
    img_name : str or None, optional
        A human-readable label for the image, used in plot titles and
        printed output. Defaults to ``"image"`` when ``None``.

    Returns
    -------
    None
        This function produces side effects only (console output and plot
        display); it does not return a value.
    """
    laterality = info["laterality"]
    top_percent = info["top_percent"]
    is_paddle = info["is_paddle"]

    label = img_name if img_name is not None else "image"
    status = "DETECTED" if is_paddle else "NOT DETECTED"
    title_color = "green" if is_paddle else "red"

    print("\n" + "=" * 90)
    print("COMPRESSION PADDLE DETECTION RESULTS")
    print("=" * 90)
    print(f"{label:<35} | Lat: {laterality} | {status}")
    print(f"Decision: {info['decision_reason']}")
    print(f"Original size: {info['original_shape'][1]}x{info['original_shape'][0]}")
    print(f"Downsampled size: {info['downsampled_shape'][1]}x{info['downsampled_shape'][0]}")
    print(f"Top brightest pixels: {top_percent}%")
    print(f"Search band x-range: {info['search_band']}")

    if info.get("vertical_count") is not None:
        print(f"Vertical check: count={info['vertical_count']} at col={info['vertical_col']}")

    print("\nRegion details:")
    for region in info["regions"]:
        print(f"  - {region['region']}: {region['reason']}")
        if region["found_edges"]:
            print(f"      edges: left={region['left_edge']}, right={region['right_edge']}")
            print(f"      left_segments={region['left_segments']}")
            print(f"      right_segments={region['right_segments']}")
            if "min_span" in region:
                print(f"      min_span={region['min_span']}, min_required={region['min_required']:.1f}")
            if "center_diff" in region:
                print(f"      center_diff={region['center_diff']:.1f}, max={region['max_center_diff']:.1f}")

    annot_orig, annot_bright = annotate_compression_paddle(
        gray_img,
        laterality=laterality,
        top_percent=top_percent,
        downsample_width=info["downsample_width"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(annot_orig)
    axes[0].set_title(
        f"Original\n{label}\nLat: {laterality} | Paddle: {is_paddle}",
        color=title_color,
        fontweight="bold",
        fontsize=10,
    )
    axes[0].axis("off")

    axes[1].imshow(annot_bright)
    axes[1].set_title(
        f"Top {top_percent}% Brightest\n{label}\nLat: {laterality} | Paddle: {is_paddle}",
        color=title_color,
        fontweight="bold",
        fontsize=10,
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def detect_compression_paddle(
    gray_img: Union[np.ndarray, Image.Image],
    laterality: Union[str, object],
    downsample_width: int = 400,
    top_percent: float = 7,
    debug: bool = False,
    img_name: Optional[str] = None,
    return_info: bool = False,
    min_width_fraction: float = 0.5,
) -> Union[bool, dict]:
    """
    Detect whether a mammogram image contains a compression paddle.

    Simple notebook usage
    ---------------------
    is_paddle = detect_compression_paddle(
        compression_img,
        laterality=lat,
        top_percent=top_percent,
        debug=True,
        img_name=img_name,
    )

    Parameters
    ----------
    gray_img : np.ndarray or PIL.Image
        Input mammogram image.
    laterality : str
        'L' or 'R'.
    downsample_width : int
        Width used internally for faster detection.
    top_percent : float
        Percent of brightest pixels to keep.
    debug : bool
        If True, prints diagnostics and shows the annotated plots.
    img_name : str or None
        Optional image name used in debug titles/prints.
    return_info : bool
        If True, return the full info dictionary instead of only bool.

    Returns
    -------
    bool or dict
        By default returns True/False.
        If return_info=True, returns a dictionary with detector details.
    """
    info = _analyze_compression_paddle(
        gray_img,
        laterality=laterality,
        downsample_width=downsample_width,
        top_percent=top_percent,
        min_width_fraction=min_width_fraction
    )

    if debug:
        _show_compression_paddle_debug(gray_img, info, img_name=img_name)

    if return_info:
        return info

    return bool(info["is_paddle"])
