part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Smooths [FaceMeshResult] landmarks across consecutive frames with the
/// official MediaPipe OneEuro configuration.
///
/// This is the [FaceMeshResult]-level wrapper around
/// [OneEuroLandmarksSmoother]. [FaceMeshInferencePipeline] applies it
/// automatically when created with `landmarkSmoothing`; use it directly when
/// driving [FaceMeshProcessor] yourself:
///
/// ```dart
/// final FaceLandmarkSmoother smoother = FaceLandmarkSmoother();
/// final FaceMeshResult smoothed = smoother.smooth(
///   rawResult,
///   timestamp: frameTimestamp,
/// );
/// ```
///
/// Use one instance per face and strictly increasing timestamps. The
/// smoothed result keeps the input's ROI and score; only the landmarks (and
/// the triangles derived from them) change.
class FaceLandmarkSmoother {
  /// Creates a smoother; [options] defaults to the official FaceLandmarker
  /// stream-mode configuration.
  FaceLandmarkSmoother({
    LandmarkSmoothingOptions options = const LandmarkSmoothingOptions(),
  }) : _smoother = OneEuroLandmarksSmoother(options: options);

  final OneEuroLandmarksSmoother _smoother;
  int? _lastImageWidth;
  int? _lastImageHeight;

  /// Drops all filter state; the next [smooth] starts a fresh sequence.
  void reset() {
    _smoother.reset();
  }

  /// Returns [result] with its landmarks smoothed against the previously
  /// seen frames.
  ///
  /// [timestamp] must be strictly increasing across calls. Filter state
  /// resets automatically when the image size or landmark count changes.
  FaceMeshResult smooth(FaceMeshResult result, {required Duration timestamp}) {
    if (result.landmarks.isEmpty ||
        result.imageWidth <= 0 ||
        result.imageHeight <= 0) {
      return result;
    }
    if (result.imageWidth != _lastImageWidth ||
        result.imageHeight != _lastImageHeight) {
      _smoother.reset();
      _lastImageWidth = result.imageWidth;
      _lastImageHeight = result.imageHeight;
    }
    final List<SmoothablePoint> smoothed = _smoother.apply(
      landmarks: <SmoothablePoint>[
        for (final FaceMeshLandmark landmark in result.landmarks)
          (x: landmark.x, y: landmark.y, z: landmark.z),
      ],
      imageWidth: result.imageWidth,
      imageHeight: result.imageHeight,
      timestamp: timestamp,
    );
    return FaceMeshResult(
      landmarks: <FaceMeshLandmark>[
        for (final SmoothablePoint point in smoothed)
          FaceMeshLandmark(x: point.x, y: point.y, z: point.z),
      ],
      rect: result.rect,
      score: result.score,
      imageWidth: result.imageWidth,
      imageHeight: result.imageHeight,
    );
  }
}
