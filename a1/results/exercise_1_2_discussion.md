# Exercise 1.2: Box Detection Discussion

## Result visualization

The implementation from exercise 1.1 detects the floor plane first, removes the floor from the valid
point cloud, and then detects the dominant top plane of the box. The generated result figures show:

- the amplitude image,
- the distance image,
- the detected floor mask,
- the detected box-top mask,
- an overlay of floor, box top, and estimated box corners,
- a subsampled 3D point cloud with the detected corners.

Generated result figures:

- `example1kinect_exercise_1_1.png`
- `example2kinect_exercise_1_1.png`
- `example3kinect_exercise_1_1.png`
- `example4kinect_exercise_1_1.png`

For the first example, the estimated dimensions are:

| Example | Height | Length | Width |
| --- | ---: | ---: | ---: |
| example1kinect.mat | 0.1961 m | 0.6331 m | 0.4569 m |
| example2kinect.mat | 0.1947 m | 0.4525 m | 0.3633 m |
| example3kinect.mat | 0.1925 m | 0.4555 m | 0.3615 m |
| example4kinect.mat | 0.1877 m | 0.4448 m | 0.3401 m |

The height is computed as the distance between the fitted floor plane and the fitted top plane. The
length and width are estimated from the 3D points inside the largest connected component of the box
top mask by projecting those points into the local coordinate system of the detected top plane.

## Algorithm summary

The point cloud contains invalid measurements, so points with `z == 0` are ignored. RANSAC samples
three valid 3D points, estimates a plane in normal form, and counts all points whose point-to-plane
distance is below a threshold. After the best plane is found, the plane is refined with SVD using the
inlier points.

The floor mask is cleaned with morphological closing/opening. The box candidate region is then formed
from non-floor points. To avoid selecting background or image-border artifacts, the implementation keeps
the largest connected non-border component and also uses the amplitude image to prefer the brighter box
surface. A second RANSAC run estimates the top plane of the box. The resulting top mask is cleaned and
its largest connected component is used for dimension estimation.

## Weaknesses

The implementation depends on several manually chosen parameters, especially the RANSAC distance
thresholds, the number of iterations, the morphology kernel size, and the amplitude threshold. These
parameters work for the provided examples, but they may fail if the sensor noise, distance to the box,
box material, or illumination changes.

RANSAC can select the wrong plane if a large background surface has more inliers than the object of
interest. This is especially likely near image borders or when background objects are planar. The current
implementation reduces this problem by removing border-connected components and using amplitude, but
that is still a heuristic.

The method assumes that the box top and the floor are approximately planar. If the box is damaged,
partially occluded, glossy, transparent, or strongly tilted, the plane fit and the corner detection can
become inaccurate.

The corner estimation is based on the detected top-mask component. If the mask has missing parts,
extra attached regions, or noisy edges, the length and width can be biased. Morphological filtering helps,
but it can also remove thin valid regions or enlarge the mask slightly.

The algorithm uses the amplitude image to help isolate the box. This is useful for these examples because
the box is relatively bright, but it is not generally safe: a dark box, bright floor, or changed exposure
could break this assumption.

RANSAC is computationally more expensive than a direct least-squares fit because many random models are
tested. Runtime is still acceptable for these images, but larger point clouds or higher iteration counts
would make it slower.

## Possible improvements

The thresholds could be chosen adaptively from the data. For example, the point-to-plane threshold could
be derived from the local depth noise or from the distribution of residuals after an initial fit, instead
of being fixed manually.

The RANSAC model selection could include additional constraints. Since the top of the box and the floor
are expected to be roughly parallel, the second plane could be required to have a normal vector close to
the floor normal. This would reduce false detections from vertical sides or background structures.

A better candidate selection step could combine distance, amplitude, connected components, and surface
normal consistency. Estimating local normals before RANSAC would help reject points that do not belong
to the same physical surface.

Instead of basic bounding-box corner estimation, the top component could be measured with a robust 2D
oriented bounding rectangle in the top-plane coordinate system. This could be combined with outlier
removal or convex-hull simplification to make length and width estimates less sensitive to noisy mask
boundaries.

The floor and box planes could be refined after RANSAC with weighted least squares. Points with lower
amplitude or larger residuals could receive smaller weights, improving accuracy in noisy areas.

Runtime could be improved by subsampling candidate points during RANSAC and then evaluating/refining the
best model on the full-resolution point cloud. Another option is early stopping when the inlier ratio is
high enough.

Finally, a confidence score should be reported together with the dimensions. Useful indicators would be
the number of inliers, residual statistics, connected-component size, and the angle between the floor and
top plane normals. This would make it easier to detect unreliable results automatically.
