# Geometry Measurements

This document describes the measurement logic exposed from
`lib/src/face_mesh_geometry.dart`.

The primary implementation is a native opt-in estimator based on MediaPipe Face
Geometry data:

- `canonical_face_model.obj` for 468-landmark results
- `face_model_with_iris.obj` for 478-landmark iris results
- `geometry_pipeline_metadata_*_landmarks.pbtxt` Procrustes weights

There is no Dart fallback metric estimator. If the native geometry symbol is
unavailable or the native solve fails, `estimateGeometry()` throws instead of
returning an arbitrary centimeter estimate.

Centimeter values are still estimates, and physical measurement quality depends
on virtual camera assumptions and optional calibration.

## Inputs

The estimator uses `FaceMeshResult.landmarks`.

Each landmark has:

```text
x = normalized image x
y = normalized image y
z = MediaPipe relative depth, approximately in image-width units
```

`estimateGeometry()` accepts 468-landmark face mesh results and 478-landmark
iris results.

When iris is enabled, the result has 478 landmarks:

```text
468..472 = one iris
473..477 = the other iris
```

## 2D Distances

### Pixel Distance

`distancePixels(a, b)` converts normalized coordinates into pixels:

```text
dx = (landmark[a].x - landmark[b].x) * imageWidth
dy = (landmark[a].y - landmark[b].y) * imageHeight

distancePx = sqrt(dx^2 + dy^2)
```

This changes when the face moves closer to or farther from the camera.

## Native Metric Coordinate Estimate

When native geometry is available, `estimateGeometry()` calls:

```text
mp_face_geometry_estimate(...)
```

The native implementation ports the MediaPipe Face Geometry screen-to-metric
pipeline shape without adding the full MediaPipe/protobuf runtime:

```text
screen landmarks
-> project x/y at the virtual camera near plane
-> flip handedness for the first weighted Procrustes scale estimate
-> move/rescale z, unproject x/y, flip handedness
-> second weighted Procrustes scale estimate
-> move/rescale z with the combined scale, unproject x/y, flip handedness
-> weighted Procrustes pose transform
-> inverse-pose metric landmarks
```

The default virtual camera options are:

```text
vertical_fov_degrees = 63.0
near = 1.0
far = 10000.0
origin = top-left
```

The Procrustes basis uses the official MediaPipe landmark weights from the
metadata files. The canonical XYZ coordinates are in centimeter units.

## 3D Distance

`geometry.distanceCm(a, b)` is a 3D straight-line distance:

```text
dx = metric[a].xCm - metric[b].xCm
dy = metric[a].yCm - metric[b].yCm
dz = metric[a].zCm - metric[b].zCm

distanceCm = sqrt(dx^2 + dy^2 + dz^2)
```

This is not a surface distance along facial curvature. It is the direct
Euclidean distance between two estimated 3D landmark points.

For face-surface distances, a separate path/polyline distance should be added,
for example:

```text
pathDistanceCm([234, ..., 454])
```

## Preset Measurements

Current measurement presets are direct 3D straight-line distances:

```text
faceWidth              = 234 <-> 454
faceHeight             = 10  <-> 152
eyeOuterDistance       = 33  <-> 263
eyeInnerDistance       = 133 <-> 362
interpupillaryDistance = 468 <-> 473 when iris exists, null otherwise
mouthWidth             = 61  <-> 291
noseWidth              = 98  <-> 327
```

`faceWidth` is cheek/outer-face width near the cheekbone line, not a contour
distance around the facial surface.

## Head Pose Estimate

Head pose is extracted from the native canonical-to-runtime pose transform
matrix.

The native pose matrix is a row-major 4x4 similarity transform:

```text
[ r00, r01, r02, tx,
  r10, r11, r12, ty,
  r20, r21, r22, tz,
  0,   0,   0,   1  ]
```

The uniform scale is removed from the 3x3 rotation part before extracting
Euler-like angles. With `rXY` referring to the normalized 3x3 rotation matrix:

```text
pitch = asin(clamp(-r12, -1, 1))
yaw   = atan2(r02, r22)
roll  = atan2(r10, r11)

degrees = radians * 180 / pi
```

Meanings:

```text
yaw   = left/right head turn
pitch = up/down head tilt
roll  = in-plane head tilt
```

## Pose Matrix

The pose matrix comes from the weighted Procrustes solve and maps the canonical
face model into runtime metric space.

The returned matrix is row-major 4x4:

```text
[ r00, r01, r02, tx,
  r10, r11, r12, ty,
  r20, r21, r22, tz,
  0,   0,   0,   1  ]
```

## Limitations

- Centimeter values are estimates.
- Native geometry depends on virtual camera assumptions unless real camera
  intrinsics are provided in a future API.
- `distanceCm()` is a 3D straight-line distance, not a face-surface contour
  distance.
