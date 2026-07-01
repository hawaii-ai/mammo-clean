import cv2
import numpy as np

from PIL import Image
from typing import Optional, Union

try:
    from IPython.display import display
except ImportError:
    display = None


def downsample_image(
    gray_img: np.ndarray,
    target_width: int = 400,
) -> tuple[np.ndarray, float]:
    """
    Args:
        gray_img: A 2-D grayscale image array of shape (H, W).
        target_width: Desired output width in pixels. Defaults to 400.

    Returns:
        A tuple of:
            - downsampled: The resized grayscale image of shape
              (target_height, target_width).
            - scale: The scale factor applied (target_width / original_width).
    """
    H, W = gray_img.shape
    scale = target_width / W
    target_height = int(H * scale)
    downsampled = cv2.resize(
        gray_img,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return downsampled, scale


def get_roi_bounds(W: int, laterality: str) -> tuple[int, int]:
    """
    Args:
        W: Total image width in pixels.
        laterality: Breast laterality — either ``'R'`` (right) or ``'L'``
            (left).

    Returns:
        A tuple ``(x_min, x_max)`` describing the inclusive/exclusive
        horizontal pixel range of the ROI column.
    """
    one_third_w = W // 3

    if laterality == "R":
        x_min, x_max = W - one_third_w, W
    else:
        x_min, x_max = 0, one_third_w

    return (x_min, x_max)


def extract_highest_intensity_pixels(
    enhanced: np.ndarray,
    top_percent: float = 1,
    intensity_threshold: Optional[float] = None,
    morph_kernel_size: int = 15,
    debug: bool = False,
) -> np.ndarray:
    """
    Args:
        enhanced: 2-D uint8 image after CLAHE.
        top_percent: Fraction of the brightest pixels to retain, expressed as a
            percentage (e.g., ``1`` keeps the top 1 %). Defaults to 1.
        intensity_threshold: Optional minimum absolute intensity value.  When
            provided the effective threshold becomes
            ``max(percentile_threshold, intensity_threshold)``.
        morph_kernel_size: Side length (in pixels) of the elliptical structuring
            element used for morphological closing. Defaults to 15.
        debug: If ``True``, print extraction statistics to stdout.

    Returns:
        A uint8 binary mask (same shape as ``enhanced``) where 255 indicates a
        retained high-intensity pixel and 0 indicates background.
    """
    percentile = 100 - top_percent
    threshold = np.percentile(enhanced, percentile)

    if intensity_threshold is not None:
        threshold = max(threshold, intensity_threshold)

    high_intensity_mask = np.zeros_like(enhanced)
    high_intensity_mask[enhanced >= threshold] = 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    high_intensity_mask = cv2.morphologyEx(
        high_intensity_mask, cv2.MORPH_CLOSE, close_kernel, iterations=3
    )

    if debug:
        high_intensity_vals = enhanced[high_intensity_mask > 0]
        if high_intensity_vals.size > 0:
            print(f"Extraction stats:")
            print(f"Threshold (top {top_percent}%): {threshold:.2f}")
            if intensity_threshold is not None:
                print(f"Intensity threshold applied: {intensity_threshold}")
            print(f"Min intensity: {high_intensity_vals.min()}")
            print(f"Max intensity: {high_intensity_vals.max()}")
            print(f"Mean intensity: {high_intensity_vals.mean():.2f}")
            print(f"Median intensity: {np.median(high_intensity_vals):.2f}")
            print(f"Number of pixels: {high_intensity_vals.size}")

    return high_intensity_mask


def analyze_connected_components(
    mask: np.ndarray,
    min_area: int = 70_000,
    max_area: int = 500_000,
    debug: bool = False,
    original_img: Optional[np.ndarray] = None,
) -> list[dict]:
    """
    Args:
        mask: A uint8 binary mask (values 0 or 255) to analyse.
        min_area: Minimum component area in pixels to retain. Defaults to
            70 000.
        max_area: Maximum component area in pixels to retain. Defaults to
            500 000.
        debug: If ``True``, print per-component statistics and display an
            annotated image with surviving component outlines drawn in green.
        original_img: Optional grayscale or colour image used as the background
            for debug visualisation.  When ``None`` the mask itself is used.

    Returns:
        A list of dicts, one per surviving component, each containing:

        - ``'label'`` (int): Connected-component label index.
        - ``'area'`` (int): Number of pixels in the component.
        - ``'bbox'`` (tuple[int, int, int, int]): ``(x, y, w, h)`` bounding box.
        - ``'centroid'`` (tuple[float, float]): ``(cx, cy)`` centroid coordinates.
        - ``'circularity'`` (float): ``4π·area / perimeter²`` (1.0 = perfect circle).
        - ``'aspect_ratio'`` (float): ``max(w, h) / min(w, h)``.
        - ``'perimeter'`` (float): Contour perimeter in pixels.
        - ``'width'`` (int): Bounding-box width.
        - ``'height'`` (int): Bounding-box height.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    components: list[dict] = []
    all_contours: list[np.ndarray] = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]

        if area < min_area or area > max_area:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        perimeter = cv2.arcLength(contours[0], closed=True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

        bbox = (x, y, w, h)
        components.append(
            {
                "label": label,
                "area": area,
                "bbox": bbox,
                "centroid": (cx, cy),
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "perimeter": perimeter,
                "width": w,
                "height": h,
            }
        )
        all_contours.append(contours[0])

    if debug:
        if components:
            print(f"Found {len(components)} components:")
            for comp in components:
                print(
                    f"Label {comp['label']}: area={comp['area']}, "
                    f"circularity={comp['circularity']:.3f}, "
                    f"aspect_ratio={comp['aspect_ratio']:.2f}"
                )

        if original_img is not None:
            vis = (
                cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                if original_img.ndim == 2
                else original_img.copy()
            )
        else:
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(vis, all_contours, contourIdx=-1, color=(0, 255, 0), thickness=4)
        Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).show()

    return components


def filter_cardiac_device_candidates(
    components: list[dict],
    circularity_min: float = 0.4,
    circularity_max: float = 0.6,
    aspect_ratio_max: float = 3.0,
    density_max: float = 0.9,
    density_min: float = 0.3,
    debug: bool = False,
) -> list[dict]:
    """
    Args:
        components: List of component dicts as returned by
            :func:`analyze_connected_components`.
        circularity_min: Lower bound on component circularity. Defaults to 0.4.
        circularity_max: Upper bound on component circularity. Defaults to 0.6.
        aspect_ratio_max: Maximum allowed aspect ratio (long / short side).
            Defaults to 3.0.
        density_max: Maximum allowed pixel density within the bounding box.
            Defaults to 0.9.
        density_min: Minimum required pixel density within the bounding box.
            Defaults to 0.3.
        debug: If ``True``, print the reason each rejected component failed and
            summarise the retained candidates.

    Returns:
        A filtered list of component dicts that pass all cardiac-device criteria.
    """
    candidates: list[dict] = []

    for comp in components:
        if not (circularity_min <= comp["circularity"] <= circularity_max):
            if debug:
                print("failed circularity", comp["circularity"])
            continue

        if comp["aspect_ratio"] > aspect_ratio_max:
            if debug:
                print("failed aspect ratio", comp["aspect_ratio"])
            continue

        bbox_area = comp["width"] * comp["height"]
        density = comp["area"] / bbox_area if bbox_area > 0 else 0

        if not (density_min <= density <= density_max):
            if debug:
                print(f"Rejected: density={density:.3f} outside [{density_min}, {density_max}]")
            continue

        candidates.append(comp)

    if debug:
        if candidates:
            print(f"{len(candidates)} components passed device filtering:")
            for comp in candidates:
                bbox_area = comp["width"] * comp["height"]
                density = comp["area"] / bbox_area
                print(
                    f"Area: {comp['area']}, Circularity: {comp['circularity']:.3f}, "
                    f"Aspect ratio: {comp['aspect_ratio']:.2f}, Density: {density:.3f}"
                )
        else:
            print("No components passed device filtering")

    return candidates


def mask_diagonal_chest_wall(img: np.ndarray, laterality: str) -> np.ndarray:
    """
    Zero-out the image region that does not contain the chest wall.

    Args:
        img: Input image as a NumPy array (grayscale 2-D or colour 3-D).
        laterality: Breast laterality — ``'L'`` (left) or ``'R'`` (right).
            Leading/trailing whitespace is stripped and the value is
            case-normalised internally.

    Returns:
        A copy of ``img`` with pixels outside the chest-wall triangle set to 0.

    Raises:
        ValueError: If ``laterality`` is ``None`` or not ``'L'`` / ``'R'``.
    """
    if laterality is None:
        raise ValueError("laterality must be provided ('L' or 'R').")

    lat = laterality.strip().upper()
    if lat not in ("L", "R"):
        raise ValueError(f"laterality must be 'L' or 'R', got: {laterality!r}")

    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if lat == "L":
        pts = np.array(
            [
                [0, 0],
                [2 * W // 3, 0], 
                [0, H], 
            ],
            dtype=np.int32,
        )
    else:
        pts = np.array(
            [
                [W // 3, 0],
                [W, 0],
                [W, H],
            ],
            dtype=np.int32,
        )

    cv2.fillConvexPoly(mask, pts, 255)
    return cv2.bitwise_and(img, img, mask=mask)


def detect_cardiac_device_in_image(
    img_input: Union[str, Image.Image, np.ndarray],
    clahe: cv2.CLAHE,
    laterality: str,
    top_percent: float = 10,
    intensity_threshold: Optional[float] = None,
    min_area: int = 70_000,
    max_area: int = 500_000,
    circularity_max: float = 0.6,
    circularity_min: float = 0.05,
    aspect_ratio_max: float = 8.0,
    density_min: float = 0.5,
    density_max: float = 0.9,
    morph_kernel_size: int = 15,
    debug: bool = False,
) -> tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[list[dict]], Optional[tuple[int, int]]]:
    """
    Args:
        img_input: The input image supplied as one of:

            * A file-path string (loaded with ``cv2.IMREAD_GRAYSCALE``).
            * A :class:`PIL.Image.Image` object (converted to grayscale).
            * A NumPy array (2-D grayscale, 3-channel RGB, or 4-channel RGBA).

        clahe: A pre-configured :class:`cv2.CLAHE` instance used for contrast
            enhancement.
        laterality: Breast laterality — ``'L'`` (left) or ``'R'`` (right) —
            used to select the correct chest-wall triangle.
        top_percent: Percentage of the brightest pixels to consider when
            building the high-intensity mask. Defaults to 10.
        intensity_threshold: Optional absolute intensity floor applied on top
            of the percentile-based threshold. Defaults to ``None``.
        min_area: Minimum connected-component area (in full-resolution pixels)
            to consider. Defaults to 70 000.
        max_area: Maximum connected-component area (in full-resolution pixels)
            to consider. Defaults to 500 000.
        circularity_max: Upper circularity bound for the device filter.
            Defaults to 0.6.
        circularity_min: Lower circularity bound for the device filter.
            Defaults to 0.05.
        aspect_ratio_max: Maximum aspect ratio for the device filter.
            Defaults to 8.0.
        density_min: Minimum pixel density (area / bbox area) for the device
            filter. Defaults to 0.5.
        density_max: Maximum pixel density (area / bbox area) for the device
            filter. Defaults to 0.9.
        morph_kernel_size: Kernel size for the morphological closing step
            inside :func:`extract_highest_intensity_pixels`. Defaults to 15.
        debug: If ``True``, display intermediate images and print diagnostic
            output at each pipeline stage.

    Returns:
        A 5-tuple:

        - ``isCardiac`` (int): ``1`` if a cardiac device was detected,
          ``0`` otherwise.
        - ``mask_fullres`` (np.ndarray | None): Full-resolution uint8 binary
          mask of the high-intensity region, or ``None`` if detection failed
          early.
        - ``img0`` (np.ndarray | None): The chest-wall-masked grayscale image
          at original resolution, or ``None`` if the image could not be loaded.
        - ``device_candidates`` (list[dict] | None): List of component dicts
          that passed the cardiac-device filter, or ``None`` if no candidates
          were found.
        - ``original_shape`` (tuple[int, int] | None): ``(H, W)`` of the
          original image, or ``None`` if the image could not be loaded.
    """
    if isinstance(img_input, str):
        img0 = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            return 0, None, None, None, None

    elif isinstance(img_input, Image.Image):
        img0 = np.array(img_input.convert("L"))

    elif isinstance(img_input, np.ndarray):
        img0 = img_input.copy()

        if img0.ndim == 3:
            if img0.shape[2] == 4:
                img0 = cv2.cvtColor(img0, cv2.COLOR_RGBA2GRAY)
            elif img0.shape[2] == 3:
                img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)

        if img0.dtype != np.uint8:
            img0 = cv2.normalize(img0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    else:
        raise TypeError("img_input must be a file path, PIL Image, or numpy array.")

    H_orig, W_orig = img0.shape

    img0 = mask_diagonal_chest_wall(img0, laterality)

    img_small, scale = downsample_image(img0, target_width=400)

    if debug:
        print(
            f"Image dimensions: {W_orig}x{H_orig} "
            f"-> {img_small.shape[1]}x{img_small.shape[0]} "
            f"(scale: {scale:.3f})"
        )

    enhanced = clahe.apply(img_small)

    if debug:
        Image.fromarray(enhanced).show()

    high_intensity_mask = extract_highest_intensity_pixels(
        enhanced,
        top_percent=top_percent,
        intensity_threshold=intensity_threshold,
        morph_kernel_size=morph_kernel_size,
        debug=debug,
    )

    if debug:
        Image.fromarray(high_intensity_mask).show()
        breast_vis = cv2.bitwise_and(enhanced, enhanced, mask=high_intensity_mask)
        display(Image.fromarray(breast_vis))

    if high_intensity_mask is None:
        return 0, None, img0, None, None

    high_intensity_vals = enhanced[high_intensity_mask > 0]
    if high_intensity_vals.size > 0:
        if np.median(high_intensity_vals) < 220:
            return 0, None, img0, None, None

    mask_fullres = cv2.resize(
        high_intensity_mask,
        (W_orig, H_orig),
        interpolation=cv2.INTER_NEAREST,
    )

    breast_vis = cv2.bitwise_and(img0, img0, mask=mask_fullres)

    components = analyze_connected_components(
        mask_fullres,
        min_area=min_area,
        max_area=max_area,
        debug=debug,
        original_img=breast_vis,
    )

    device_candidates = filter_cardiac_device_candidates(
        components,
        circularity_min=circularity_min,
        circularity_max=circularity_max,
        density_min=density_min,
        density_max=density_max,
        aspect_ratio_max=aspect_ratio_max,
        debug=debug,
    )

    if len(components) > 2:
        isCardiac = 0
    else:
        isCardiac = 1 if device_candidates else 0

    return isCardiac, mask_fullres, img0, device_candidates, (H_orig, W_orig)