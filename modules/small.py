import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from typing import List, Tuple


def annotate_small_breast(
    gray_img: np.ndarray,
    edge_strip_px: int = 15,
) -> Image.Image:
    """
    Args:
        gray_img: 2-D uint8 NumPy array containing the grayscale mammogram.
        edge_strip_px: Height in pixels of the top and bottom strips sampled
            inside the centre band. Defaults to 15.

    Returns:
        Annotated PIL ``Image`` in RGB mode with the yellow band outline and
        red/blue segment lines drawn on it.
    """
    H, W = gray_img.shape
    x0, x1, y0, y1 = _center_band_shrunk(W, H, shrink_ratio=0.2)

    # Extract top and bottom strips from the yellow band
    top_strip = gray_img[y0:y0 + edge_strip_px, x0:x1]
    bot_strip = gray_img[y1 - edge_strip_px:y1, x0:x1]

    top_mask = top_strip > 0
    bot_mask = bot_strip > 0

    top_has_intensity = np.any(top_mask)
    bot_has_intensity = np.any(bot_mask)

    rgb = Image.new('RGB', (W, H))
    rgb.paste(Image.fromarray(gray_img))
    draw = ImageDraw.Draw(rgb)
    draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 255, 0), width=5)

    rgb.show()

    # Process top strip: find leftmost and rightmost x coordinates with intensity
    if top_has_intensity:
        top_mask_any = np.any(top_mask, axis=0)  # shape: (x1-x0,)
        top_segments = _get_horizontal_segments(top_mask_any, x0)

        # Draw each segment separately (no bridging across gaps)
        for left_x, right_x in top_segments:
            top_y = y0 + edge_strip_px // 2  # mid of strip; purely for visualization
            draw.line([(left_x, top_y), (right_x, top_y)], fill=(255, 0, 0), width=5)
            draw.line([(left_x, top_y), (left_x, H)], fill=(255, 0, 0), width=5)
            draw.line([(right_x, top_y), (right_x, H)], fill=(255, 0, 0), width=5)

        print(f"Top strip: red line from x={left_x} to x={right_x} at y={top_y}")

        rgb.show()

    # Process bottom strip: find leftmost and rightmost x coordinates with intensity
    if bot_has_intensity:
        bot_mask_any = np.any(bot_mask, axis=0)
        bot_segments = _get_horizontal_segments(bot_mask_any, x0)

        for left_x, right_x in bot_segments:
            bot_y = y1 - edge_strip_px // 2
            draw.line([(left_x, bot_y), (right_x, bot_y)], fill=(0, 0, 255), width=5)
            draw.line([(left_x, bot_y), (left_x, 0)], fill=(0, 0, 255), width=5)
            draw.line([(right_x, bot_y), (right_x, 0)], fill=(0, 0, 255), width=5)

        print(f"Bottom strip: blue line from x={left_x} to x={right_x} at y={bot_y}")

        rgb.show()

    return rgb


def _get_horizontal_segments(
    mask_row: np.ndarray,
    x0: int,
    gap_tolerance: int = 3,
) -> List[Tuple[int, int]]:
    """
    Args:
        mask_row: 1-D boolean array of length ``W_strip`` (the width of the
            centre band), where ``True`` indicates a bright pixel.
        x0: Global x-coordinate of the left edge of the centre band. Added
            to all segment endpoints so callers receive image-space coordinates.
        gap_tolerance: Maximum number of consecutive dark pixels allowed
            inside a segment before it is split into two. Defaults to 3.

    Returns:
        List of ``(left_x, right_x)`` tuples in global image coordinates,
        one per detected contiguous segment.  Returns an empty list if no
        bright pixels are found.
    """
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    last_bright = -gap_tolerance - 1

    for x in range(mask_row.shape[0]):
        if mask_row[x]:
            gap_from_last = x - last_bright

            if not in_seg:
                in_seg = True
                start = x
            elif gap_from_last > gap_tolerance:
                segments.append((start + x0, last_bright + x0))
                start = x

            last_bright = x

    if in_seg:
        segments.append((start + x0, last_bright + x0))

    return segments


def _center_band_shrunk(
    W: int,
    H: int,
    shrink_ratio: float = 0.2,
) -> Tuple[int, int, int, int]:
    """
    Args:
        W: Image width in pixels.
        H: Image height in pixels.
        shrink_ratio: Fraction of the base band width to remove (split
            equally between the left and right insets). Defaults to 0.2.

    Returns:
        A tuple ``(x0, x1, y0, y1)`` where:
            - ``x0``: Left column of the shrunk centre band (inclusive).
            - ``x1``: Right column of the shrunk centre band (exclusive).
            - ``y0``: Top row (always 0).
            - ``y1``: Bottom row (always ``H``).
    """
    base_x0 = W // 3
    base_x1 = 2 * W // 3
    base_w = base_x1 - base_x0
    shrink = int(shrink_ratio * base_w)
    x0 = base_x0 + shrink // 2
    x1 = base_x1 - shrink // 2
    return x0, x1, 0, H


def detect_small_breast_compression_paddle(
    gray_img: np.ndarray,
    edge_strip_px: int = 25,
    max_span_percent: float = 0.5,
    min_span_pixels: int = 5,
    gap_tolerance: int = 3,
    max_span_ratio: float = 0.85,
    max_segments: int = 6,
    debug: bool = False,
) -> int:
    """
    Detect the presence of a small breast compression paddle in a mammogram.

    Args:
        gray_img: Grayscale mammogram as a 2-D uint8 NumPy array or any
            array-like that ``np.array()`` can convert to a 2-D array.
        edge_strip_px: Height in pixels of the top and bottom strips sampled
            inside the centre band. Defaults to 25.
        max_span_percent: Maximum allowed segment span as a fraction of the
            centre-band width. Segments wider than this are treated as breast
            edges and discarded. Defaults to 0.5.
        min_span_pixels: Minimum allowed segment span in pixels. Segments
            narrower than this are treated as thin artefacts and discarded.
            Defaults to 5.
        gap_tolerance: Maximum dark-pixel gap bridged within a single segment
            when calling ``_get_horizontal_segments``. Defaults to 3.
        max_span_ratio: Maximum allowed span asymmetry between the top and
            bottom segments of the chosen pair, computed as
            ``(max_span − min_span) / max_span``. Pairs exceeding this ratio
            are rejected. Defaults to 0.85.
        max_segments: Maximum number of contiguous segments permitted in either
            strip before the image is rejected as too complex. Defaults to 6.
        debug: If ``True``, print diagnostic information and display an
            annotated visualisation of the centre band. Defaults to ``False``.

    Returns:
        ``1`` if a compression paddle was detected, ``0`` otherwise.
    """
    gray_img = np.array(gray_img)
    H, W = gray_img.shape
    x0, x1, y0, y1 = _center_band_shrunk(W, H, shrink_ratio=0.2)

    top_strip = gray_img[y0:y0 + edge_strip_px, x0:x1]
    bot_strip = gray_img[y1 - edge_strip_px:y1, x0:x1]

    top_mask = top_strip > 0
    bot_mask = bot_strip > 0

    if debug:
        annot = annotate_small_breast(gray_img)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(np.array(annot))
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    if not (np.any(top_mask) and np.any(bot_mask)):
        return 0

    yellow_box_width = x1 - x0
    max_span = int(max_span_percent * yellow_box_width)

    # Collapse to 1D masks over x
    top_any = np.any(top_mask, axis=0)
    bot_any = np.any(bot_mask, axis=0)

    top_segments = _get_horizontal_segments(top_any, x0, gap_tolerance=gap_tolerance)
    bot_segments = _get_horizontal_segments(bot_any, x0, gap_tolerance=gap_tolerance)

    if debug:
        print("Top segments:", top_segments)
        print("Bot segments:", bot_segments)

    if len(top_segments) > max_segments or len(bot_segments) > max_segments:
        if debug:
            print(f"REJECTED: Too many segments (top={len(top_segments)}, bot={len(bot_segments)}, max={max_segments})")
        return 0

    # Filter by span: BOTH minimum (reject thin artifacts) AND maximum (reject wide breast edges)
    top_segments = [s for s in top_segments if min_span_pixels <= (s[1] - s[0]) <= max_span]
    bot_segments = [s for s in bot_segments if min_span_pixels <= (s[1] - s[0]) <= max_span]

    if debug:
        print("Filtered top segments:", top_segments)
        print("Filtered bot segments:", bot_segments)

    if not top_segments or not bot_segments:
        return 0

    def segments_overlap(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> bool:
        """Return ``True`` if two ``(left, right)`` segments share any x range."""
        left1, right1 = seg1
        left2, right2 = seg2
        return not (right1 < left2 or right2 < left1)

    # Find all overlapping pairs
    overlapping_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for top_seg in top_segments:
        for bot_seg in bot_segments:
            if segments_overlap(top_seg, bot_seg):
                overlapping_pairs.append((top_seg, bot_seg))

    if debug:
        print("Overlapping pairs:", overlapping_pairs)

    if not overlapping_pairs:
        if debug:
            print("No overlapping segment pairs found")
        return 0

    def overlap_quality(pair: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """Return the pixel width of the overlapping region for a segment pair."""
        top_seg, bot_seg = pair
        left1, right1 = top_seg
        left2, right2 = bot_seg
        overlap_start = max(left1, left2)
        overlap_end = min(right1, right2)
        overlap_width = overlap_end - overlap_start
        return overlap_width

    best_pair = max(overlapping_pairs, key=overlap_quality)
    top_seg, bot_seg = best_pair

    top_left, top_right = top_seg
    bot_left, bot_right = bot_seg

    top_span = top_right - top_left
    bot_span = bot_right - bot_left

    max_span = max(top_span, bot_span)
    min_span = min(top_span, bot_span)

    if max_span > 0:
        span_ratio = (max_span - min_span) / max_span

        if debug:
            print(f"Top span: {top_span}, Bottom span: {bot_span}")
            print(f"Span ratio: {span_ratio:.2f} (max allowed: {max_span_ratio})")

        if span_ratio > max_span_ratio:
            if debug:
                print(f"REJECTED: Spans too different (ratio={span_ratio:.2f} > {max_span_ratio})")
            return 0

    if debug:
        print("Chosen pair:", best_pair)
        print("DETECTED")

    return 1