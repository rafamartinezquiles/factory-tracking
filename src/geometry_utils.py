from __future__ import annotations

from typing import Dict, Iterable, Tuple

# Basic type aliases for clarity
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def bbox_center(box: BBox) -> Point:
    """
    Return the (x, y) center of a bounding box.
    """
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_height(box: BBox) -> float:
    """
    Return the height of a bounding box in pixels.
    """
    _, y1, _, y2 = box
    return y2 - y1


def bbox_foot_point(box: BBox) -> Point:
    """
    Approximate a player's foot position as the midpoint of the bottom edge
    of the bounding box. This gives a stable ground reference for tracking.
    """
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), y2)


def euclidean_distance(p1: Point, p2: Point) -> float:
    """
    Compute straight-line Euclidean distance between two points.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def axis_distances(p1: Point, p2: Point) -> Tuple[float, float]:
    """
    Return absolute per-axis differences between two points.
    Useful when structural alignment (separate dx, dy) is needed.
    """
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def convert_pixels_to_meters(
    pixel_distance: float,
    reference_height_m: float,
    reference_height_px: float,
) -> float:
    """
    Convert a pixel distance into meters based on the ratio between
    reference real-world height and observed pixel height.
    """
    if reference_height_px == 0:
        return 0.0
    return (pixel_distance * reference_height_m) / reference_height_px


def convert_meters_to_pixels(
    meters: float,
    reference_height_m: float,
    reference_height_px: float,
) -> float:
    """
    Convert a distance in meters into pixel units using a known mapping
    between reference physical height and pixel height.
    """
    if reference_height_m == 0:
        return 0.0
    return (meters * reference_height_px) / reference_height_m


def closest_keypoint_index(
    point: Point,
    flat_keypoints: Iterable[float],
    candidate_indices: Iterable[int],
) -> int:
    """
    Among a set of candidate court keypoints, return the index whose
    vertical (y-axis) distance to the given point is minimal.
    """
    best_idx = None
    best_distance = float("inf")

    for kp_idx in candidate_indices:
        kp_y = flat_keypoints[kp_idx * 2 + 1]
        distance = abs(point[1] - kp_y)
        if distance < best_distance:
            best_distance = distance
            best_idx = kp_idx

    # Fallback in case no candidate exists
    if best_idx is None:
        best_idx = list(candidate_indices)[0]

    return best_idx
