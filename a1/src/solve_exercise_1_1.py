"""Exercise 1.1 - box detection."""

import os
from pathlib import Path

import matplotlib

# This avoids some Matplotlib cache problems on my setup.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

# I only need to save the figure.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import ndimage


# settings

EXAMPLE_FILE = Path("../data/example1kinect.mat")
OUTPUT_DIR = Path("../results")

FLOOR_THRESHOLD = 0.004
TOP_THRESHOLD = 0.008
RANSAC_ITERATIONS = 900


# load data

def load_example(path):
    """Load the example file."""

    mat = scipy.io.loadmat(path)

    # The names in the files are amplitudes1, distances1, cloud1, etc.
    amplitude_key = next(key for key in mat if key.startswith("amplitudes"))
    distance_key = next(key for key in mat if key.startswith("distances"))
    cloud_key = next(key for key in mat if key.startswith("cloud"))

    amplitude = mat[amplitude_key]
    distance = mat[distance_key]
    cloud = mat[cloud_key]

    return amplitude, distance, cloud


# plane model for RANSAC

def fit_plane_from_three_points(points):
    """Fit one plane from three points."""

    p0, p1, p2 = points

    # two directions on the plane
    v1 = p1 - p0
    v2 = p2 - p0

    # normal vector of the plane
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    # bad random sample
    if norm < 1e-10:
        return None

    normal = normal / norm
    offset = float(np.dot(normal, p0))

    # same sign convention each time
    if offset < 0:
        normal = -normal
        offset = -offset

    return normal, offset


def point_to_plane_distances(points, plane):
    """Distances from points to a plane."""

    normal, offset = plane
    return np.abs(points @ normal - offset)


def refine_plane_with_all_inliers(points):
    """Fit the plane again with the inliers."""

    center = points.mean(axis=0)

    # last SVD direction is used as the plane normal
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    offset = float(np.dot(normal, center))

    if offset < 0:
        normal = -normal
        offset = -offset

    return normal, offset


# RANSAC

def ransac_plane(cloud, valid_mask, threshold, max_iterations, random_seed):
    """Small RANSAC implementation for one plane."""

    # remove invalid points
    valid_mask = valid_mask & np.isfinite(cloud).all(axis=2) & (cloud[:, :, 2] > 0)

    valid_indices = np.argwhere(valid_mask)
    points = cloud[valid_mask]

    if len(points) < 3:
        raise ValueError("Need at least three valid points to estimate a plane.")

    rng = np.random.default_rng(random_seed)

    best_plane = None
    best_inliers = np.zeros(len(points), dtype=bool)

    for _ in range(max_iterations):
        # three points make one plane candidate
        sample_ids = rng.choice(len(points), size=3, replace=False)
        plane = fit_plane_from_three_points(points[sample_ids])

        if plane is None:
            continue

        # count points that fit this plane
        inliers = point_to_plane_distances(points, plane) < threshold

        if inliers.sum() > best_inliers.sum():
            best_plane = plane
            best_inliers = inliers

            # best possible case
            if inliers.all():
                break

    if best_plane is None:
        raise RuntimeError("RANSAC could not find a valid plane.")

    # fit again using all inliers
    refined_plane = refine_plane_with_all_inliers(points[best_inliers])
    final_inliers = point_to_plane_distances(points, refined_plane) < threshold

    # back to image shape
    inlier_mask = np.zeros(cloud.shape[:2], dtype=bool)
    inlier_mask[valid_indices[final_inliers, 0], valid_indices[final_inliers, 1]] = True

    return refined_plane, inlier_mask


# filtering the masks

def clean_mask(mask, iterations=2, fill_holes=False):
    """Simple mask cleanup."""

    structure = np.ones((5, 5), dtype=bool)

    # closing/opening makes the mask less noisy
    cleaned = ndimage.binary_closing(mask, structure=structure, iterations=iterations)
    cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=1)

    if fill_holes:
        cleaned = ndimage.binary_fill_holes(cleaned)

    return cleaned


def largest_component(mask):
    """Keep the largest connected region."""

    labels, count = ndimage.label(mask)

    if count == 0:
        return mask & False

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    return labels == sizes.argmax()


def largest_non_border_component(mask):
    """Keep a large component that is not attached to the border."""

    labels, count = ndimage.label(mask)

    if count == 0:
        return mask & False

    border_labels = set(labels[0, :])
    border_labels.update(labels[-1, :])
    border_labels.update(labels[:, 0])
    border_labels.update(labels[:, -1])

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    # usually the border regions are background
    for label in border_labels:
        sizes[label] = 0

    # fallback
    if sizes.max() == 0:
        return largest_component(mask)

    return labels == sizes.argmax()


# measuring the box

def top_plane_box_geometry(cloud, top_mask):
    """Estimate corners and size from the top mask."""

    points = cloud[top_mask]
    center = points.mean(axis=0)

    # two main directions on the top plane
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    basis = vh[:2]

    # coordinates in the top plane
    projected = (points - center) @ basis.T

    min_u, min_v = projected.min(axis=0)
    max_u, max_v = projected.max(axis=0)

    # rectangle around the projected points
    corners_uv = np.array(
        [
            [min_u, min_v],
            [max_u, min_v],
            [max_u, max_v],
            [min_u, max_v],
        ]
    )

    # back to 3D
    corners_xyz = center + corners_uv @ basis

    # nearest pixels for drawing the corners on the image
    valid_rc = np.argwhere(top_mask)
    valid_xyz = cloud[top_mask]

    corner_pixels = []
    for corner in corners_xyz:
        nearest = np.argmin(np.linalg.norm(valid_xyz - corner, axis=1))
        corner_pixels.append(valid_rc[nearest])

    length, width = sorted([max_u - min_u, max_v - min_v], reverse=True)

    return np.asarray(corner_pixels), corners_xyz, float(length), float(width)


# visualization

def save_visualization(amplitude, distance, cloud, floor_mask, top_mask, corners_rc,
                       corners_xyz, height, length, width, output_path):
    """Save the result figure."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    axes[0].imshow(amplitude, cmap="gray")
    axes[0].set_title("Amplitude")

    axes[1].imshow(distance, cmap="viridis")
    axes[1].set_title("Distance")

    axes[2].imshow(floor_mask, cmap="gray")
    axes[2].set_title("Floor mask")

    axes[3].imshow(top_mask, cmap="gray")
    axes[3].set_title("Box top mask")

    # blue = floor, orange = box top
    overlay = np.zeros((*top_mask.shape, 3), dtype=float)
    overlay[floor_mask] = [0.0, 0.35, 1.0]
    overlay[top_mask] = [1.0, 0.25, 0.0]

    axes[4].imshow(distance, cmap="gray")
    axes[4].imshow(overlay, alpha=0.55)
    axes[4].plot(corners_rc[:, 1], corners_rc[:, 0], "yo-", linewidth=2)
    axes[4].set_title("Floor, top, and corners")

    # only plot every few points
    valid = np.isfinite(cloud).all(axis=2) & (cloud[:, :, 2] > 0)
    step = 5
    sampled = cloud[::step, ::step]
    sampled_valid = valid[::step, ::step]

    axes[5].remove()
    ax3d = fig.add_subplot(2, 3, 6, projection="3d")

    xyz = sampled[sampled_valid]
    ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=0.3)
    ax3d.scatter(corners_xyz[:, 0], corners_xyz[:, 1], corners_xyz[:, 2], c="red", s=40)

    ax3d.set_title("Subsampled point cloud")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    for ax in axes[:5]:
        ax.axis("off")

    fig.suptitle(f"height={height:.3f} m, length={length:.3f} m, width={width:.3f} m")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# run exercise

amplitude, distance, cloud = load_example(EXAMPLE_FILE)

# valid 3D points
valid = np.isfinite(cloud).all(axis=2) & (cloud[:, :, 2] > 0)

# find floor
floor_plane, floor_raw = ransac_plane(
    cloud=cloud,
    valid_mask=valid,
    threshold=FLOOR_THRESHOLD,
    max_iterations=RANSAC_ITERATIONS,
    random_seed=11,
)

floor_mask = clean_mask(floor_raw, iterations=1, fill_holes=False)

# remove floor
candidate_mask = valid & ~floor_mask

# box is brighter in the amplitude image
amplitude_values = amplitude[candidate_mask]
if amplitude_values.size:
    bright_threshold = np.percentile(amplitude_values, 55)
    candidate_mask = candidate_mask & (amplitude >= bright_threshold)

# remove large border/background parts
candidate_mask = largest_non_border_component(candidate_mask)

# find box top
top_plane, top_raw = ransac_plane(
    cloud=cloud,
    valid_mask=candidate_mask,
    threshold=TOP_THRESHOLD,
    max_iterations=RANSAC_ITERATIONS,
    random_seed=23,
)

top_mask = clean_mask(top_raw & candidate_mask, iterations=1, fill_holes=True)
top_mask = largest_component(top_mask)

# measure dimensions
floor_normal, floor_offset = floor_plane
top_normal, top_offset = top_plane

height = abs(top_offset - floor_offset)
corners_rc, corners_xyz, length, width = top_plane_box_geometry(cloud, top_mask)

# save results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

stem = EXAMPLE_FILE.stem
image_path = OUTPUT_DIR / f"{stem}_exercise_1_1.png"
summary_path = OUTPUT_DIR / f"{stem}_exercise_1_1_summary.txt"

save_visualization(
    amplitude=amplitude,
    distance=distance,
    cloud=cloud,
    floor_mask=floor_mask,
    top_mask=top_mask,
    corners_rc=corners_rc,
    corners_xyz=corners_xyz,
    height=height,
    length=length,
    width=width,
    output_path=image_path,
)

summary = [
    f"Example: {EXAMPLE_FILE}",
    f"Floor plane: n={np.array2string(floor_normal, precision=5)}, d={floor_offset:.5f}",
    f"Top plane:   n={np.array2string(top_normal, precision=5)}, d={top_offset:.5f}",
    f"Height: {height:.4f} m",
    f"Length: {length:.4f} m",
    f"Width:  {width:.4f} m",
    "Corners (row, col):",
    np.array2string(corners_rc),
    "Corners (x, y, z):",
    np.array2string(corners_xyz, precision=5),
]

summary_path.write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary))
