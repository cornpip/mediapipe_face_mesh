import 'dart:async';
import 'dart:ui' show Offset, Rect, Size;

import 'mediapipe_face_mesh.dart';

typedef FaceMeshBoxResolver<T> = FaceMeshBox? Function(T frame);

/// Helper that turns a stream of camera frames into MediaPipe results.
class FaceMeshStreamProcessor {
  FaceMeshStreamProcessor(this._processor);

  final FaceMeshProcessor _processor;

  /// Processes a stream of [FaceMeshImage] frames sequentially.
  ///
  /// Provide either a static [roi] or a [boxResolver] callback to define the
  /// region of interest per frame (not both). Every incoming frame is processed
  /// with the same parameters that you would pass to [FaceMeshProcessor.process].
  Stream<FaceMeshResult> processImages(
    Stream<FaceMeshImage> frames, {
    NormalizedRect? roi,
    FaceMeshBoxResolver<FaceMeshImage>? boxResolver,
    double boxScale = 1.2,
    bool boxMakeSquare = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) async* {
    _validateResolvers<FaceMeshImage>(roi, boxResolver);
    await for (final FaceMeshImage frame in frames) {
      final FaceMeshBox? dynamicBox = boxResolver?.call(frame);
      yield _processor.process(
        frame,
        roi: roi,
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
    FaceMeshBoxResolver<FaceMeshNv21Image>? boxResolver,
    double boxScale = 1.0,
    bool boxMakeSquare = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) async* {
    _validateResolvers<FaceMeshNv21Image>(roi, boxResolver);
    await for (final FaceMeshNv21Image frame in frames) {
      final FaceMeshBox? dynamicBox = boxResolver?.call(frame);
      yield _processor.processNv21(
        frame,
        roi: roi,
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
    FaceMeshBoxResolver<T>? boxResolver,
  ) {
    if (roi != null && boxResolver != null) {
      throw ArgumentError('Provide either roi or boxResolver, not both.');
    }
  }
}

/// Creates a stream processor for camera/image frames.
FaceMeshStreamProcessor createFaceMeshStreamProcessor(
  FaceMeshProcessor processor,
) =>
    FaceMeshStreamProcessor(processor);

/// Returns the face bounding box in pixel coordinates.
Rect faceMeshBoundingRect(
  FaceMeshResult result, {
  Size? targetSize,
  bool clampToBounds = true,
}) =>
    result.boundingRect(
      targetSize: targetSize,
      clampToBounds: clampToBounds,
    );

/// Converts a landmark to a pixel-space [Offset].
Offset faceMeshLandmarkOffset(
  FaceMeshResult result,
  FaceMeshLandmark landmark, {
  Size? targetSize,
  bool clampToBounds = true,
}) =>
    result.landmarkAsOffset(
      landmark,
      targetSize: targetSize,
      clampToBounds: clampToBounds,
    );

/// Converts all landmarks into pixel-space [Offset]s.
List<Offset> faceMeshLandmarksOffsets(
  FaceMeshResult result, {
  Size? targetSize,
  bool clampToBounds = true,
}) =>
    result.landmarksAsOffsets(
      targetSize: targetSize,
      clampToBounds: clampToBounds,
    );
