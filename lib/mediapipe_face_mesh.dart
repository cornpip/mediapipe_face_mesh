import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart' as pkg_ffi;
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/src/mediapipe_face_bindings_generated.dart';
import 'src/native_bindings_loader.dart';

part 'src/native_converters.dart';

part 'src/face_mesh_utils.dart';

part 'src/face_mesh_result_utils.dart';

const String _defaultModelAsset =
    'packages/mediapipe_face_mesh/assets/models/mediapipe_face_mesh.tflite';

final Finalizer<ffi.Pointer<MpFaceMeshContext>> _contextFinalizer =
    Finalizer<ffi.Pointer<MpFaceMeshContext>>(
      (pointer) => faceBindings.mp_face_mesh_destroy(pointer),
    );

/// Integer constants describing the pixel formats understood by the native side.
class FaceMeshPixelFormat {
  const FaceMeshPixelFormat._();

  /// RGBA (red, green, blue, alpha) ordering expected by MediaPipe.
  static const int rgba = 0;

  /// BGRA ordering for buffers that come directly from some platforms.
  static const int bgra = 1;
}

/// Delegate types supported by the native runtime.
enum FaceMeshDelegate {
  /// Execute on the built-in CPU interpreter.
  cpu,

  /// Use the XNNPACK delegate when available.
  xnnpack,

  /// Use the GPU delegate (V2) when supported by the runtime.
  gpuV2,
}

/// Immutable normalized rectangle that MediaPipe uses as ROI input.
class NormalizedRect {
  /// Builds a normalized rectangle from center, size, and rotation.
  const NormalizedRect({
    required this.xCenter,
    required this.yCenter,
    required this.width,
    required this.height,
    this.rotation = 0,
  });

  /// X coordinate of the rectangle center in normalized space (0..1).
  final double xCenter;

  /// Y coordinate of the rectangle center in normalized space (0..1).
  final double yCenter;

  /// Rectangle width as a fraction of the image width.
  final double width;

  /// Rectangle height as a fraction of the image height.
  final double height;

  /// Clockwise rotation in radians.
  final double rotation;

  /// Creates a rectangle using the native MediaPipe layout.
  factory NormalizedRect.fromNative(MpNormalizedRect rect) => NormalizedRect(
    xCenter: rect.x_center,
    yCenter: rect.y_center,
    width: rect.width,
    height: rect.height,
    rotation: rect.rotation,
  );
}

/// Pixel-space bounding box used to derive a normalized ROI.
///
/// You can use this helper when providing bounding regions to
/// [FaceMeshProcessor.process] or [FaceMeshProcessor.processNv21].
class FaceMeshBox {
  /// Creates a pixel bounding box from explicit edges.
  const FaceMeshBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  /// Convenience for building a box from top-left/width/height coordinates.
  factory FaceMeshBox.fromLTWH({
    required double left,
    required double top,
    required double width,
    required double height,
  }) => FaceMeshBox(
    left: left,
    top: top,
    right: left + width,
    bottom: top + height,
  );

  /// Left coordinate in pixels.
  final double left;

  /// Top coordinate in pixels.
  final double top;

  /// Right coordinate in pixels.
  final double right;

  /// Bottom coordinate in pixels.
  final double bottom;

  /// Width of the rectangle in pixels.
  double get width => right - left;

  /// Height of the rectangle in pixels.
  double get height => bottom - top;

  /// Horizontal center of the rectangle.
  double get centerX => (left + right) * 0.5;

  /// Vertical center of the rectangle.
  double get centerY => (top + bottom) * 0.5;
}

/// Container that holds RGBA/BGRA pixels used as inference input.
class FaceMeshImage {
  /// Creates an RGBA/BGRA image wrapper from raw bytes.
  FaceMeshImage({
    required this.pixels,
    required this.width,
    required this.height,
    this.pixelFormat = FaceMeshPixelFormat.rgba,
    int? bytesPerRow,
  }) : bytesPerRow = bytesPerRow ?? width * 4 {
    final int requiredBytes = this.bytesPerRow * height;
    if (pixels.length < requiredBytes) {
      throw ArgumentError(
        'Pixel buffer is smaller than required size ($requiredBytes bytes).',
      );
    }
    if (pixelFormat != FaceMeshPixelFormat.rgba &&
        pixelFormat != FaceMeshPixelFormat.bgra) {
      throw ArgumentError('Unsupported pixel format: $pixelFormat');
    }
  }

  /// Raw pixel buffer backing this image.
  final Uint8List pixels;

  /// Frame width in pixels.
  final int width;

  /// Frame height in pixels.
  final int height;

  /// Bytes consumed per row (stride).
  final int bytesPerRow;

  /// Pixel format understood by the native layer.
  final int pixelFormat;
}

/// Holder for NV21 (Y + interleaved VU) camera buffers.
class FaceMeshNv21Image {
  /// Creates an NV21 image from Y and interleaved VU planes.
  FaceMeshNv21Image({
    required this.yPlane,
    required this.vuPlane,
    required this.width,
    required this.height,
    int? yBytesPerRow,
    int? vuBytesPerRow,
  }) : yBytesPerRow = yBytesPerRow ?? width,
       vuBytesPerRow = vuBytesPerRow ?? width {
    if (width <= 0 || height <= 0) {
      throw ArgumentError('Invalid image size: ${width}x$height');
    }
    final int requiredY = this.yBytesPerRow * height;
    final int requiredVu = this.vuBytesPerRow * (height ~/ 2);
    if (yPlane.length < requiredY) {
      throw ArgumentError('Y plane buffer too small (need $requiredY bytes).');
    }
    if (vuPlane.length < requiredVu) {
      throw ArgumentError(
        'VU plane buffer too small (need $requiredVu bytes).',
      );
    }
    if ((height & 1) != 0) {
      throw ArgumentError('NV21 height must be even.');
    }
  }

  /// Luma plane (full resolution).
  final Uint8List yPlane;

  /// Interleaved VU chroma plane.
  final Uint8List vuPlane;

  /// Frame width in pixels.
  final int width;

  /// Frame height in pixels (must be even).
  final int height;

  /// Row stride for the Y plane.
  final int yBytesPerRow;

  /// Row stride for the VU plane.
  final int vuBytesPerRow;
}

/// A single 3D landmark returned by MediaPipe.
class FaceMeshLandmark {
  /// Builds a landmark from normalized coordinates returned by MediaPipe.
  FaceMeshLandmark({required this.x, required this.y, required this.z});

  /// Horizontal coordinate normalized to [0, 1].
  final double x;

  /// Vertical coordinate normalized to [0, 1].
  final double y;

  /// Depth relative to the camera in canonical MediaPipe units.
  final double z;
}

/// Aggregates the results of a single face mesh inference.
class FaceMeshResult {
  /// Constructs a result using landmark points, ROI and scores.
  FaceMeshResult({
    required this.landmarks,
    required this.rect,
    required this.score,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// All face landmarks returned by the native graph.
  final List<FaceMeshLandmark> landmarks;

  /// Normalized rectangle covering the detected face.
  final NormalizedRect rect;

  /// Confidence score reported by MediaPipe.
  final double score;

  /// Width of the image used during inference.
  final int imageWidth;

  /// Height of the image used during inference.
  final int imageHeight;
}

/// Base exception thrown by this plugin when native calls fail.
class MediapipeFaceMeshException implements Exception {
  /// Creates an exception with a human-readable [message].
  MediapipeFaceMeshException(this.message);

  /// Cause string returned by the native layer.
  final String message;

  @override
  String toString() => 'MediapipeFaceMeshException($message)';
}

/// High-level wrapper around the native MediaPipe Face Mesh graph.
class FaceMeshProcessor {
  FaceMeshProcessor._(this._context) {
    _contextFinalizer.attach(this, _context, detach: this);
  }

  static const double _boxScale = 1.2;

  final ffi.Pointer<MpFaceMeshContext> _context;
  bool _closed = false;

  /// Creates the native interpreter and loads a model.
  static Future<FaceMeshProcessor> create({
    int threads = 2,
    double minDetectionConfidence = 0.5,
    double minTrackingConfidence = 0.5,
    bool enableSmoothing = true,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
  }) async {
    final String resolvedModelPath = await _materializeModel();

    final optionsPtr = pkg_ffi.calloc<MpFaceMeshCreateOptions>();
    final ffi.Pointer<pkg_ffi.Utf8> modelPathPtr = resolvedModelPath
        .toNativeUtf8();
    try {
      optionsPtr.ref
        ..threads = threads
        ..min_detection_confidence = minDetectionConfidence
        ..min_tracking_confidence = minTrackingConfidence
        ..delegate = delegate.index
        ..enable_smoothing = enableSmoothing ? 1 : 0
        ..tflite_library_path = ffi.nullptr;

      final ffi.Pointer<MpFaceMeshContext> context = faceBindings
          .mp_face_mesh_create(modelPathPtr.cast(), optionsPtr);
      if (context == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_mesh_last_global_error()) ??
              'Failed to create face mesh context.',
        );
      }
      return FaceMeshProcessor._(context);
    } finally {
      pkg_ffi.calloc.free(optionsPtr);
      pkg_ffi.malloc.free(modelPathPtr);
    }
  }

  /// Processes an image and returns face landmarks.
  ///
  /// By default, this processes the full frame. To restrict processing to a
  /// region, provide either:
  /// - [roi] as a normalized rectangle, or
  /// - [box] as a pixel-space bounding box (converted to an ROI internally).
  ///
  /// When [box] is provided, it is converted into a square ROI by default
  /// (using the max of width/height) and optionally expanded by [boxScale].
  FaceMeshResult process(
    FaceMeshImage image, {
    NormalizedRect? roi,
    FaceMeshBox? box,
    double boxScale = _boxScale,
    bool boxMakeSquare = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    _ensureNotClosed();
    if (roi != null && box != null) {
      throw ArgumentError('Provide either roi or box, not both.');
    }
    if (rotationDegrees != 0 &&
        rotationDegrees != 90 &&
        rotationDegrees != 180 &&
        rotationDegrees != 270) {
      throw ArgumentError('rotationDegrees must be one of {0, 90, 180, 270}.');
    }
    final int logicalWidth = (rotationDegrees == 90 || rotationDegrees == 270)
        ? image.height
        : image.width;
    final int logicalHeight = (rotationDegrees == 90 || rotationDegrees == 270)
        ? image.width
        : image.height;
    final NormalizedRect? effectiveRoi =
        roi ??
        (box != null
            ? _normalizedRectFromBox(
                box,
                imageWidth: logicalWidth,
                imageHeight: logicalHeight,
                scale: boxScale,
                makeSquare: boxMakeSquare,
              )
            : null);
    final _NativeImage nativeImage = _toNativeImage(image);
    final ffi.Pointer<MpNormalizedRect> roiPtr = effectiveRoi != null
        ? _toNativeRect(effectiveRoi)
        : ffi.nullptr;
    FaceMeshResult? processed;
    try {
      final ffi.Pointer<MpFaceMeshResult> resultPtr = faceBindings
          .mp_face_mesh_process(
            _context,
            nativeImage.image,
            roiPtr == ffi.nullptr ? ffi.nullptr : roiPtr,
            rotationDegrees,
            mirrorHorizontal ? 1 : 0,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_mesh_last_error(_context)) ??
              'Native face mesh error.',
        );
      }
      processed = _copyResult(resultPtr.ref);
      faceBindings.mp_face_mesh_release_result(resultPtr);
    } finally {
      pkg_ffi.calloc.free(nativeImage.pixels);
      pkg_ffi.calloc.free(nativeImage.image);
      if (roiPtr != ffi.nullptr) {
        pkg_ffi.calloc.free(roiPtr);
      }
    }
    return processed;
  }

  /// Processes NV21 camera frames captured directly from a camera preview.
  ///
  /// Parameters mirror the [process] method although the inputs are provided as
  /// separate Y and VU planes in NV21 layout. Set [mirrorHorizontal] to true if
  /// your camera preview is mirrored to avoid flipped outputs.
  FaceMeshResult processNv21(
    FaceMeshNv21Image image, {
    NormalizedRect? roi,
    FaceMeshBox? box,
    double boxScale = _boxScale,
    bool boxMakeSquare = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    _ensureNotClosed();
    if (roi != null && box != null) {
      throw ArgumentError('Provide either roi or box, not both.');
    }
    if (rotationDegrees != 0 &&
        rotationDegrees != 90 &&
        rotationDegrees != 180 &&
        rotationDegrees != 270) {
      throw ArgumentError('rotationDegrees must be one of {0, 90, 180, 270}.');
    }
    final int logicalWidth = (rotationDegrees == 90 || rotationDegrees == 270)
        ? image.height
        : image.width;
    final int logicalHeight = (rotationDegrees == 90 || rotationDegrees == 270)
        ? image.width
        : image.height;
    final NormalizedRect? effectiveRoi =
        roi ??
        (box != null
            ? _normalizedRectFromBox(
                box,
                imageWidth: logicalWidth,
                imageHeight: logicalHeight,
                scale: boxScale,
                makeSquare: boxMakeSquare,
              )
            : null);
    final _NativeNv21Image nativeImage = _toNativeNv21Image(image);
    final ffi.Pointer<MpNormalizedRect> roiPtr = effectiveRoi != null
        ? _toNativeRect(effectiveRoi)
        : ffi.nullptr;
    FaceMeshResult? processed;
    try {
      final ffi.Pointer<MpFaceMeshResult> resultPtr = faceBindings
          .mp_face_mesh_process_nv21(
            _context,
            nativeImage.image,
            roiPtr == ffi.nullptr ? ffi.nullptr : roiPtr,
            rotationDegrees,
            mirrorHorizontal ? 1 : 0,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_mesh_last_error(_context)) ??
              'Native face mesh error.',
        );
      }
      processed = _copyResult(resultPtr.ref);
      faceBindings.mp_face_mesh_release_result(resultPtr);
    } finally {
      pkg_ffi.calloc.free(nativeImage.yPlane);
      pkg_ffi.calloc.free(nativeImage.vuPlane);
      pkg_ffi.calloc.free(nativeImage.image);
      if (roiPtr != ffi.nullptr) {
        pkg_ffi.calloc.free(roiPtr);
      }
    }
    return processed;
  }

  FaceMeshResult _copyResult(MpFaceMeshResult nativeResult) {
    final ffi.Pointer<MpLandmark> landmarkPtr = nativeResult.landmarks;
    final List<FaceMeshLandmark> landmarks =
        (landmarkPtr == ffi.nullptr || nativeResult.landmarks_count <= 0)
        ? <FaceMeshLandmark>[]
        : List<FaceMeshLandmark>.generate(nativeResult.landmarks_count, (
            int i,
          ) {
            final MpLandmark lm = (landmarkPtr + i).ref;
            return FaceMeshLandmark(x: lm.x, y: lm.y, z: lm.z);
          });

    return FaceMeshResult(
      landmarks: landmarks,
      rect: NormalizedRect.fromNative(nativeResult.rect),
      score: nativeResult.score,
      imageWidth: nativeResult.image_width,
      imageHeight: nativeResult.image_height,
    );
  }

  /// Releases the native context and associated resources.
  void close() {
    if (_closed) {
      return;
    }
    _contextFinalizer.detach(this);
    faceBindings.mp_face_mesh_destroy(_context);
    _closed = true;
  }

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Face mesh context already closed.');
    }
  }
}
