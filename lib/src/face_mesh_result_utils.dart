part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Pixel-space helpers attached to [FaceMeshResult] instances.
extension FaceMeshResultPixels on FaceMeshResult {
  /// Returns the face bounding box in pixel coordinates.
  ///
  /// [targetSize] defaults to the dimensions observed during inference.
  /// When [clampToBounds] is true, the rectangle is limited to the provided
  /// size; set it to false to preserve out-of-frame values.
  Rect boundingRect({Size? targetSize, bool clampToBounds = true}) {
    final Size size = targetSize ?? _inferenceSize;
    double clampX(double value) =>
        clampToBounds ? value.clamp(0.0, size.width) : value;
    double clampY(double value) =>
        clampToBounds ? value.clamp(0.0, size.height) : value;

    final double rawCenterX = rect.xCenter * size.width;
    final double rawCenterY = rect.yCenter * size.height;
    final double centerX = clampToBounds
        ? rawCenterX.clamp(0.0, size.width)
        : rawCenterX;
    final double centerY = clampToBounds
        ? rawCenterY.clamp(0.0, size.height)
        : rawCenterY;
    final double halfWidth = rect.width * size.width * 0.5;
    final double halfHeight = rect.height * size.height * 0.5;

    final double left = clampX(centerX - halfWidth);
    final double top = clampY(centerY - halfHeight);
    final double right = clampX(centerX + halfWidth);
    final double bottom = clampY(centerY + halfHeight);

    return Rect.fromLTRB(left, top, right, bottom);
  }

  /// Converts a landmark to a pixel-space [Offset].
  Offset landmarkAsOffset(
    FaceMeshLandmark landmark, {
    Size? targetSize,
    bool clampToBounds = true,
  }) {
    final Size size = targetSize ?? _inferenceSize;
    double scaleX(double value) =>
        (clampToBounds ? value.clamp(0.0, 1.0) : value) * size.width;
    double scaleY(double value) =>
        (clampToBounds ? value.clamp(0.0, 1.0) : value) * size.height;
    return Offset(scaleX(landmark.x), scaleY(landmark.y));
  }

  /// Convenience that converts all landmarks into pixel-space [Offset]s.
  List<Offset> landmarksAsOffsets({
    Size? targetSize,
    bool clampToBounds = true,
  }) {
    return landmarks
        .map(
          (FaceMeshLandmark lm) => landmarkAsOffset(
            lm,
            targetSize: targetSize,
            clampToBounds: clampToBounds,
          ),
        )
        .toList(growable: false);
  }

  Size get _inferenceSize =>
      Size(imageWidth.toDouble(), imageHeight.toDouble());
}
