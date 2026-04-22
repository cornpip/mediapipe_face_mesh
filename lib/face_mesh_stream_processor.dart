import 'dart:async';
import 'dart:ui' show Offset, Rect, Size;

import 'mediapipe_face_mesh.dart';

/// Signature used to derive a bounding box from an incoming frame.
typedef FaceMeshBoxResolver<T> = FaceMeshBox? Function(T frame);

/// Signature used to derive a normalized ROI from an incoming frame.
typedef FaceMeshRoiResolver<T> = NormalizedRect? Function(T frame);

/// Helper that turns a stream of camera frames into MediaPipe results.
class FaceMeshStreamProcessor {
  /// Creates a processor bound to the given synchronous [FaceMeshProcessor].
  FaceMeshStreamProcessor(this._processor);

  static const double _boxScale = 1.2;

  final FaceMeshProcessor _processor;

  /// Processes a stream of [FaceMeshImage] frames sequentially.
  ///
  /// Provide either a static [roi] or a [boxResolver] callback to define the
  /// region of interest per frame (not both). Every incoming frame is processed
  /// with the same parameters that you would pass to [FaceMeshProcessor.process].
  Stream<FaceMeshResult> process(
    Stream<FaceMeshImage> frames, {
    NormalizedRect? roi,
    FaceMeshRoiResolver<FaceMeshImage>? roiResolver,
    FaceMeshBoxResolver<FaceMeshImage>? boxResolver,
    double boxScale = _boxScale,
    bool boxMakeSquare = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) async* {
    _validateResolvers<FaceMeshImage>(roi, roiResolver, boxResolver);
    await for (final FaceMeshImage frame in frames) {
      final NormalizedRect? dynamicRoi = roiResolver?.call(frame);
      final FaceMeshBox? dynamicBox = boxResolver?.call(frame);
      yield _processor.process(
        frame,
        roi: dynamicRoi ?? roi,
        box: dynamicBox,
        boxScale: boxScale,
        boxMakeSquare: boxMakeSquare,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
      );
    }
  }

  /// Processes NV21 camera frames coming from a stream.
  ///
  /// The behaviour mirrors [FaceMeshProcessor.processNv21]. Provide at most one
  /// of [roi] or [boxResolver].
  Stream<FaceMeshResult> processNv21(
    Stream<FaceMeshNv21Image> frames, {
    NormalizedRect? roi,
    FaceMeshRoiResolver<FaceMeshNv21Image>? roiResolver,
    FaceMeshBoxResolver<FaceMeshNv21Image>? boxResolver,
    double boxScale = _boxScale,
    bool boxMakeSquare = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) async* {
    _validateResolvers<FaceMeshNv21Image>(roi, roiResolver, boxResolver);
    await for (final FaceMeshNv21Image frame in frames) {
      final NormalizedRect? dynamicRoi = roiResolver?.call(frame);
      final FaceMeshBox? dynamicBox = boxResolver?.call(frame);
      yield _processor.processNv21(
        frame,
        roi: dynamicRoi ?? roi,
        box: dynamicBox,
        boxScale: boxScale,
        boxMakeSquare: boxMakeSquare,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
      );
    }
  }

  void _validateResolvers<T>(
    NormalizedRect? roi,
    FaceMeshRoiResolver<T>? roiResolver,
    FaceMeshBoxResolver<T>? boxResolver,
  ) {
    if ((roi != null && roiResolver != null) ||
        (roi != null && boxResolver != null) ||
        (roiResolver != null && boxResolver != null)) {
      throw ArgumentError(
        'Provide only one of roi, roiResolver, or boxResolver.',
      );
    }
  }
}

/// Creates a stream processor for camera/image frames.
FaceMeshStreamProcessor createFaceMeshStreamProcessor(
  FaceMeshProcessor processor,
) => FaceMeshStreamProcessor(processor);

/// Returns the face bounding box in pixel coordinates.
Rect faceMeshBoundingRect(
  FaceMeshResult result, {
  Size? targetSize,
  bool clampToBounds = true,
}) => result.boundingRect(targetSize: targetSize, clampToBounds: clampToBounds);

/// Converts a landmark to a pixel-space [Offset].
Offset faceMeshLandmarkOffset(
  FaceMeshResult result,
  FaceMeshLandmark landmark, {
  Size? targetSize,
  bool clampToBounds = true,
}) => result.landmarkAsOffset(
  landmark,
  targetSize: targetSize,
  clampToBounds: clampToBounds,
);

/// Converts all landmarks into pixel-space [Offset]s.
List<Offset> faceMeshLandmarksOffsets(
  FaceMeshResult result, {
  Size? targetSize,
  bool clampToBounds = true,
}) => result.landmarksAsOffsets(
  targetSize: targetSize,
  clampToBounds: clampToBounds,
);
