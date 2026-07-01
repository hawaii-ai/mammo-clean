import cv2
import re
import numpy as np

from PIL import Image
from typing import Optional, Tuple, Union

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def overlay_blue_highlight(
    img_gray: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Args:
        img_gray: Grayscale input image as a 2-D uint8 NumPy array.
        mask: Binary mask (same H×W as ``img_gray``). Pixels where
            ``mask > 0`` receive the blue tint.
        alpha: Blend weight for the blue overlay in [0, 1]. 0 = no tint,
            1 = fully blue. Defaults to 0.3.

    Returns:
        3-channel (RGB) uint8 array of the same spatial dimensions as
        ``img_gray`` with the blue highlight applied.
    """
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    blue_overlay = rgb.copy()
    blue_overlay[mask > 0] = [0, 0, 255]

    highlighted = cv2.addWeighted(rgb, 1 - alpha, blue_overlay, alpha, 0)
    return highlighted


def get_implant_roi_bounds(
    W: int,
    laterality: str,
) -> Tuple[int, int]:
    """
    Args:
        W: Total image width in pixels.
        laterality: ``'R'`` for right breast or ``'L'`` for left breast.

    Returns:
        A tuple ``(x_min, x_max)`` defining the half-width column range
        ``[x_min, x_max)`` that covers the lateral side of the breast.
    """
    half_w = W // 2

    if laterality == 'R':
        x_min, x_max = half_w, W 
    else:
        x_min, x_max = 0, half_w 

    return (x_min, x_max)


def extract_breast_roi_mask(
    img_small: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Args:
        img_small: Downscaled grayscale mammogram as a 2-D uint8 NumPy array.

    Returns:
        A tuple ``(mask_filled, enhanced)`` where:
            - ``mask_filled`` is a binary uint8 mask (255 = breast, 0 = background),
              or ``None`` if no contours were detected.
            - ``enhanced`` is the CLAHE-enhanced version of ``img_small``.
    """
    enhanced = clahe.apply(img_small)
    blur = cv2.GaussianBlur(enhanced, (9, 9), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_fg, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_fg, iterations=1)

    ff = binary.copy()
    ff_pad = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(ff, ff_pad, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    filled = cv2.bitwise_or(binary, holes)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, enhanced

    areas = [cv2.contourArea(cnt) for cnt in contours]
    best_cnt = contours[np.argmax(areas)]
    mask = np.zeros_like(filled)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)

    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big_kernel)

    ff = mask_closed.copy()
    ff_pad = np.zeros((mask_closed.shape[0] + 2, mask_closed.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(ff, ff_pad, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    mask_filled = cv2.bitwise_or(mask_closed, holes)

    return mask_filled, enhanced


def extract_highest_intensity_pixels(
    enhanced: np.ndarray,
    breast_mask: np.ndarray,
    top_percent: float = 15,
    debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Args:
        enhanced: CLAHE-enhanced grayscale image 
        breast_mask: Binary breast ROI mask (uint8). Non-zero pixels define
            the breast region.
        top_percent: Percentage of the brightest breast pixels to select.
            For example, 15 selects the top 15 % by intensity.
            Defaults to 15.
        debug: If ``True``, print the computed intensity threshold and the
            number of selected pixels. Defaults to ``False``.

    Returns:
        A uint8 binary mask (255 = high-intensity breast pixel, 0 = other)
        with the same shape as ``enhanced``, or ``None`` if no breast pixels
        were found in ``breast_mask``.
    """
    inside = enhanced[breast_mask > 0]

    if inside.size == 0:
        return None

    percentile = 100 - top_percent
    threshold = np.percentile(inside, percentile)

    threshold = min(230, threshold)

    high_intensity_mask = np.zeros_like(breast_mask)
    high_intensity_mask[(enhanced >= threshold) & (breast_mask > 0)] = 255

    if debug:
        print(f"    Top {top_percent}% threshold: {threshold:.2f}")
        print(f"    Pixels extracted: {np.sum(high_intensity_mask > 0)}")

    return high_intensity_mask


def check_implant_in_middle_strip_v2(
    mask_fullres: np.ndarray,
    img_orig: np.ndarray,
    img_shape: Tuple[int, int],
    laterality: str,
    area_threshold: int = 80000,
    circularity_threshold: float = 0.35,
    largest_component_min_area: int = 60000,
    high_intensity_ratio_threshold: float = 0.40,
    high_intensity_value: int = 180,
    apply_closing: bool = True,
    top_n_components: int = 3,
    debug: bool = False,
) -> Tuple[bool, int, np.ndarray, dict]:
    """
    Classify whether a breast implant is present in the lateral middle band.

    Primary criteria (all must pass):
        - Component area  >= ``largest_component_min_area``
        - Fill density    >= 0.40 (component area / bounding-box area)

    Shape/intensity criteria (at least one must pass):
        - Override 1: high-intensity ratio >= 0.70 *and* large area.
        - Override 2: high-intensity ratio >= 0.25 *and* circularity >= 0.40.
        - Standard:   circularity >= ``circularity_threshold`` *and*
                      high-intensity ratio >= ``high_intensity_ratio_threshold``.

    Args:
        mask_fullres: Full-resolution high-intensity binary mask (uint8) with
            the same H×W as the original image.
        img_orig: Full-resolution grayscale original image (2-D uint8 array).
        img_shape: ``(H, W)`` tuple of the image dimensions.
        laterality: ``'R'`` for right breast or ``'L'`` for left breast, used
            to select the lateral ROI column range.
        area_threshold: Minimum total foreground area (pixels) within the
            middle strip to proceed with component analysis. Defaults to 80000.
        circularity_threshold: Minimum circularity score ``4π·A/P²`` for a
            component to pass the shape check. Defaults to 0.35.
        largest_component_min_area: Minimum area (pixels) for a candidate
            component to be considered at all. Defaults to 60000.
        high_intensity_ratio_threshold: Minimum fraction of component pixels
            that must exceed ``high_intensity_value``. Defaults to 0.40.
        high_intensity_value: Pixel intensity threshold used when computing the
            high-intensity ratio. Defaults to 180.
        apply_closing: If ``True``, apply one iteration of elliptical
            morphological closing (kernel 10×10) to the middle strip mask
            before component analysis. Defaults to ``True``.
        top_n_components: Number of largest connected components to evaluate.
            Defaults to 3.
        debug: If ``True``, print per-component metrics and render intermediate
            visualisations via ``display()``. Defaults to ``False``.

    Returns:
        A tuple ``(is_implant, area_in_middle, middle_strip_mask, debug_info)``
        where:
            - ``is_implant`` (bool): ``True`` if an implant was detected.
            - ``area_in_middle`` (int): Total foreground pixel count in the
              middle strip.
            - ``middle_strip_mask`` (np.ndarray): The (possibly morphologically
              closed) binary mask restricted to the middle strip ROI.
            - ``debug_info`` (dict): Diagnostic metrics including per-candidate
              area, circularity, density, intensity ratios, pass/fail flags, and
              the rank of the best matching candidate.
    """
    H, W = img_shape

    top_boundary = H // 5
    bottom_boundary = 4 * H // 5

    x_min, x_max = get_implant_roi_bounds(W, laterality)

    middle_strip_mask = np.zeros_like(mask_fullres)
    middle_strip_mask[top_boundary:bottom_boundary, x_min:x_max] = \
        mask_fullres[top_boundary:bottom_boundary, x_min:x_max]

    if debug:
        display(Image.fromarray(middle_strip_mask).resize((middle_strip_mask.shape[1] // 4, middle_strip_mask.shape[0] // 4)))

    if apply_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        middle_strip_mask = cv2.morphologyEx(middle_strip_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    area_in_middle = np.sum(middle_strip_mask > 0)

    debug_info = {
        'area': area_in_middle,
        'num_components': 0,
        'pass_area': False,
        'candidates': [],
        'best_candidate_rank': None,
    }

    if area_in_middle < area_threshold:
        if debug:
            print(f"      Total area: {area_in_middle:,} px (< {area_threshold:,}) - FAIL")
        return False, area_in_middle, middle_strip_mask, debug_info
    debug_info['pass_area'] = True

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        middle_strip_mask, connectivity=8
    )
    debug_info['num_components'] = num_labels - 1

    if num_labels <= 1:
        if debug:
            print("      No components found - FAIL")
        return False, area_in_middle, middle_strip_mask, debug_info

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(component_areas)[::-1]
    top_indices = [int(i) + 1 for i in order[:top_n_components]]

    is_implant = False
    best_candidate_rank = None

    if debug:
        print(f"      ROI: lateral middle (y=[{top_boundary}, {bottom_boundary}], x=[{x_min}, {x_max}]) LAT={laterality}")
        print(f"      Total area: {area_in_middle:,} px (>= {area_threshold:,}) - PASS")
        print(f"      Components: {num_labels - 1}, evaluating top {len(top_indices)} by area")

    for rank, idx in enumerate(top_indices, start=1):
        comp_area = int(stats[idx, cv2.CC_STAT_AREA])

        candidate_mask = np.zeros_like(middle_strip_mask)
        candidate_mask[labels == idx] = 255

        region_pixels = img_orig[candidate_mask > 0]
        if len(region_pixels) > 0:
            mean_intensity = float(np.mean(region_pixels))
            high_intensity_ratio = float(np.sum(region_pixels >= high_intensity_value) / len(region_pixels))
        else:
            mean_intensity = 0.0
            high_intensity_ratio = 0.0

        pass_largest_component = comp_area >= largest_component_min_area

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            perimeter, circularity = 0.0, 0.0
        else:
            perimeter = float(cv2.arcLength(contours[0], True))
            circularity = (4 * np.pi * comp_area) / (perimeter ** 2) if perimeter > 0 else 0.0

        bbox_area = stats[idx, cv2.CC_STAT_WIDTH] * stats[idx, cv2.CC_STAT_HEIGHT]
        density = comp_area / bbox_area if bbox_area > 0 else 0.0

        pass_circularity = circularity >= circularity_threshold
        pass_density = density >= 0.4
        pass_high_intensity_ratio = high_intensity_ratio >= high_intensity_ratio_threshold

        override_by_high_intensity = (
            high_intensity_ratio >= 0.7 and comp_area >= largest_component_min_area
        )
        override_by_moderate_intensity_good_shape = (
            high_intensity_ratio >= 0.25 and circularity >= 0.4
        )

        candidate_is_implant = (
            pass_largest_component and
            pass_density and
            (
                override_by_high_intensity or
                override_by_moderate_intensity_good_shape or
                (pass_circularity and pass_high_intensity_ratio)
            )
        )

        candidate_info = {
            'rank': rank, 'area': comp_area, 'perimeter': perimeter,
            'circularity': circularity, 'density': density,
            'mean_intensity': mean_intensity, 'high_intensity_ratio': high_intensity_ratio,
            'pass_largest_component': pass_largest_component,
            'pass_circularity': pass_circularity, 'pass_density': pass_density,
            'pass_high_intensity_ratio': pass_high_intensity_ratio,
            'override_by_high_intensity': override_by_high_intensity,
            'override_by_moderate_intensity_good_shape': override_by_moderate_intensity_good_shape,
            'classification': 'implant' if candidate_is_implant else 'no implant',
        }
        debug_info['candidates'].append(candidate_info)

        if debug:
            print(f"      --- Candidate #{rank} ---")

            candidate_overlay = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
            if contours:
                cv2.drawContours(candidate_overlay, contours, -1, (0, 255, 0), thickness=max(1, H // 300))
            display(
                Image.fromarray(candidate_overlay)
                     .resize((candidate_overlay.shape[1] // 4, candidate_overlay.shape[0] // 4))
            )

            print(f"      Area: {comp_area:,} px (>= {largest_component_min_area:,}) - "
                  f"{'PASS' if pass_largest_component else 'FAIL'}")
            print(f"      Perimeter: {perimeter:.1f} px")
            print(f"      Circularity: {circularity:.3f} (>= {circularity_threshold:.2f}) - "
                  f"{'PASS' if pass_circularity else 'FAIL'}")
            print(f"      Density: {density:.3f} (>= 0.40) - "
                  f"{'PASS' if pass_density else 'FAIL'}")
            print(f"      Mean intensity: {mean_intensity:.1f}")
            print(f"      High-intensity ratio: {high_intensity_ratio:.2%} "
                  f"(>= {high_intensity_ratio_threshold:.0%} @ >= {high_intensity_value}) - "
                  f"{'PASS' if pass_high_intensity_ratio else 'FAIL'}")
            if override_by_high_intensity:
                print("      OVERRIDE 1: HI >= 70% + large area -> implant")
            if override_by_moderate_intensity_good_shape:
                print("      OVERRIDE 2: HI >= 25% and C >= 0.40 -> implant")
            print(f"      Candidate #{rank} classification: {candidate_info['classification']}")

        if candidate_is_implant and not is_implant:
            is_implant = True
            best_candidate_rank = rank
            if not debug:
                break

    debug_info['best_candidate_rank'] = best_candidate_rank

    reference = next((c for c in debug_info['candidates'] if c['rank'] == best_candidate_rank), None)
    if reference is None and debug_info['candidates']:
        reference = debug_info['candidates'][0]
    if reference is not None:
        debug_info.update({
            'largest_component_area': reference['area'],
            'circularity': reference['circularity'],
            'density': reference['density'],
            'perimeter': reference['perimeter'],
            'mean_intensity': reference['mean_intensity'],
            'high_intensity_ratio': reference['high_intensity_ratio'],
            'pass_largest_component': reference['pass_largest_component'],
            'pass_circularity': reference['pass_circularity'],
            'pass_density': reference['pass_density'],
            'pass_high_intensity_ratio': reference['pass_high_intensity_ratio'],
            'override_by_high_intensity': reference['override_by_high_intensity'],
            'override_by_moderate_intensity_good_shape': reference['override_by_moderate_intensity_good_shape'],
        })

    if debug:
        print(f"      Final classification: {'implant' if is_implant else 'no implant'}"
              + (f" (matched candidate #{best_candidate_rank})" if best_candidate_rank else ""))

    return is_implant, area_in_middle, middle_strip_mask, debug_info


def close_edge_bounded_shapes(mask: np.ndarray) -> np.ndarray:
    """
    Args:
        mask: Input binary mask (uint8). Values > 127 are treated as foreground.

    Returns:
        uint8 binary mask (same shape as ``mask``) where edge-bounded open
        regions have been filled. Values are 0 or 255.
    """
    h, w = mask.shape[:2]

    binary = (mask > 127).astype(np.uint8) * 255
    pad = 2
    padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Flood fill background from a corner of the padded image
    flood_filled = padded.copy()
    cv2.floodFill(flood_filled, None, (0, 0), 128)

    flood_filled = flood_filled[pad:pad + h, pad:pad + w]
    enclosed = (flood_filled != 128).astype(np.uint8) * 255
    result = cv2.bitwise_or(binary, enclosed)

    return result


def detect_implant_in_image(
    img_input: Union[str, np.ndarray],
    laterality: str,
    top_percent: float = 30,
    area_threshold: int = 100000,
    circularity_threshold: float = 0.38,
    largest_component_min_area: int = 80000,
    high_intensity_ratio_threshold: float = 0.45,
    high_intensity_value: int = 190,
    debug: bool = False,
) -> Tuple[int, int, Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """
    Full implant detection pipeline for a single mammogram image.

    Args:
        img_input: Either a file-system path (str) to a grayscale mammogram
            image, or a pre-loaded grayscale NumPy array (2-D uint8).
        laterality: ``'R'`` for right breast or ``'L'`` for left breast.
        top_percent: Percentage of brightest breast pixels to extract in the
            high-intensity mask step. Defaults to 30.
        area_threshold: Minimum total foreground area (pixels) in the lateral
            middle band required before component analysis is attempted.
            Defaults to 100000.
        circularity_threshold: Minimum circularity ``4π·A/P²`` for a
            connected component to pass the standard shape check.
            Defaults to 0.38.
        largest_component_min_area: Minimum area (pixels) for an individual
            component to be evaluated as a candidate implant. Defaults to 80000.
        high_intensity_ratio_threshold: Minimum fraction of a candidate
            component's pixels that must exceed ``high_intensity_value``.
            Defaults to 0.45.
        high_intensity_value: Pixel intensity threshold used when computing
            the high-intensity ratio within each candidate component.
            Defaults to 190.
        debug: If ``True``, print intermediate metrics and render diagnostic
            visualisations via ``display()``. Defaults to ``False``.

    Returns:
        A 7-tuple ``(is_implant, area_in_middle, mask_fullres, img0,
        breast_mask_fullres, middle_strip_mask, debug_info)`` where:
            - ``is_implant`` (int): 1 if an implant was detected, 0 otherwise.
            - ``area_in_middle`` (int): Total foreground pixels in the lateral
              middle band.
            - ``mask_fullres`` (np.ndarray | None): Full-resolution binary mask
              of the top-N % brightest breast pixels after edge-closing, or
              ``None`` on early failure.
            - ``img0`` (np.ndarray | None): The original grayscale image array,
              or ``None`` if loading failed.
            - ``breast_mask_fullres`` (np.ndarray | None): Full-resolution
              breast ROI mask, or ``None`` on early failure.
            - ``middle_strip_mask`` (np.ndarray | None): Binary mask restricted
              to the lateral middle band used for component analysis, or
              ``None`` on early failure.
            - ``debug_info`` (dict | None): Diagnostic metrics dict from
              ``check_implant_in_middle_strip_v2``, or ``None`` on early
              failure.
    """
    # Accept either path OR array
    if isinstance(img_input, str):
        img0 = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
    else:
        img0 = img_input

    if img0 is None:
        return False, 0, None, None, None, None, None

    H_orig, W_orig = img0.shape

    target_size = (600, 800)
    img_small = cv2.resize(img0, target_size, interpolation=cv2.INTER_AREA)

    if debug:
        print(f"  Image dimensions: {W_orig}x{H_orig} -> {img_small.shape[1]}x{img_small.shape[0]}")

    breast_mask, enhanced = extract_breast_roi_mask(img_small)

    if breast_mask is None:
        return False, 0, None, img0, None, None, None

    high_intensity_mask = extract_highest_intensity_pixels(
        enhanced, breast_mask, top_percent=top_percent, debug=debug
    )
    if high_intensity_mask is None:
        breast_mask_fullres = cv2.resize(breast_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        return False, 0, None, img0, breast_mask_fullres, None, None

    mask_fullres = cv2.resize(high_intensity_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    breast_mask_fullres = cv2.resize(breast_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    mask_fullres = close_edge_bounded_shapes(mask_fullres)

    if debug:
        print('edge closed mask')
        breast_vis = cv2.bitwise_and(img0, img0, mask=mask_fullres)
        display(Image.fromarray(breast_vis))

    if debug:
        overlay = overlay_blue_highlight(img0, mask_fullres)
        display(Image.fromarray(overlay).resize((overlay.shape[1] // 4, overlay.shape[0] // 4)))

    is_implant, area_in_middle, middle_strip_mask, debug_info = check_implant_in_middle_strip_v2(
        mask_fullres, img0, (H_orig, W_orig),
        laterality=laterality,
        area_threshold=area_threshold,
        circularity_threshold=circularity_threshold,
        largest_component_min_area=largest_component_min_area,
        high_intensity_ratio_threshold=high_intensity_ratio_threshold,
        high_intensity_value=high_intensity_value,
        debug=debug
    )

    is_implant = int(is_implant)

    return is_implant, area_in_middle, mask_fullres, img0, breast_mask_fullres, middle_strip_mask, debug_info