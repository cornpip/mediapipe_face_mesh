import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart' as pkg_ffi;
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_bindings_generated.dart';
import 'src/bindings.dart';

part 'src/native_converters.dart';

const String _defaultModelAsset =
    'packages/mediapipe_face_mesh/assets/models/mediapipe_face_mesh.tflite';

final Finalizer<ffi.Pointer<MpFaceMeshContext>> _contextFinalizer =
    Finalizer<ffi.Pointer<MpFaceMeshContext>>(
        (pointer) => faceBindings.mp_face_mesh_destroy(pointer));

/// Represents the pixel format understood by the native preprocessor.
class FaceMeshPixelFormat {
  const FaceMeshPixelFormat._();

  static const int rgba = 0;
  static const int bgra = 1;
}

/// Immutable description of the normalized rectangle used by MediaPipe.
class NormalizedRect {
  const NormalizedRect({
    required this.xCenter,
    required this.yCenter,
    required this.width,
    required this.height,
    this.rotation = 0,
  });

  final double xCenter;
  final double yCenter;
  final double width;
  final double height;
  final double rotation;

  factory NormalizedRect.fromNative(MpNormalizedRect rect) => NormalizedRect(
        xCenter: rect.x_center,
        yCenter: rect.y_center,
        width: rect.width,
        height: rect.height,
        rotation: rect.rotation,
      );
}

/// A bounding box expressed in image pixel coordinates.
///
/// This is a convenience container used to derive a [NormalizedRect] ROI for
/// [MediapipeFaceMesh.process].
class FaceMeshBox {
  const FaceMeshBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  factory FaceMeshBox.fromLTWH({
    required double left,
    required double top,
    required double width,
    required double height,
  }) =>
      FaceMeshBox(
        left: left,
        top: top,
        right: left + width,
        bottom: top + height,
      );

  final double left;
  final double top;
  final double right;
  final double bottom;

  double get width => right - left;
  double get height => bottom - top;
  double get centerX => (left + right) * 0.5;
  double get centerY => (top + bottom) * 0.5;
}

/// Container that holds RGBA/BGRA pixels used as inference input.
class FaceMeshImage {
  FaceMeshImage({
    required this.pixels,
    required this.width,
    required this.height,
    this.pixelFormat = FaceMeshPixelFormat.rgba,
    int? bytesPerRow,
  })  : bytesPerRow = bytesPerRow ?? width * 4 {
    final int requiredBytes = this.bytesPerRow * height;
    if (pixels.length < requiredBytes) {
      throw ArgumentError(
          'Pixel buffer is smaller than required size ($requiredBytes bytes).');
    }
    if (pixelFormat != FaceMeshPixelFormat.rgba &&
        pixelFormat != FaceMeshPixelFormat.bgra) {
      throw ArgumentError('Unsupported pixel format: $pixelFormat');
    }
  }

  final Uint8List pixels;
  final int width;
  final int height;
  final int bytesPerRow;
  final int pixelFormat;
}

/// Holder for NV21 (Y + interleaved VU) camera buffers.
class FaceMeshNv21Image {
  FaceMeshNv21Image({
    required this.yPlane,
    required this.vuPlane,
    required this.width,
    required this.height,
    int? yBytesPerRow,
    int? vuBytesPerRow,
  })  : yBytesPerRow = yBytesPerRow ?? width,
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
          'VU plane buffer too small (need $requiredVu bytes).');
    }
    if ((height & 1) != 0) {
      throw ArgumentError('NV21 height must be even.');
    }
  }

  final Uint8List yPlane;
  final Uint8List vuPlane;
  final int width;
  final int height;
  final int yBytesPerRow;
  final int vuBytesPerRow;
}

class FaceMeshLandmark {
  FaceMeshLandmark({
    required this.x,
    required this.y,
    required this.z,
  });

  final double x;
  final double y;
  final double z;
}

class FaceMeshResult {
  FaceMeshResult({
    required this.landmarks,
    required this.rect,
    required this.score,
    required this.imageWidth,
    required this.imageHeight,
  });

  final List<FaceMeshLandmark> landmarks;
  final NormalizedRect rect;
  final double score;
  final int imageWidth;
  final int imageHeight;
}

class MediapipeFaceMeshException implements Exception {
  MediapipeFaceMeshException(this.message);
  final String message;

  @override
  String toString() => 'MediapipeFaceMeshException($message)';
}

class MediapipeFaceMesh {
  MediapipeFaceMesh._(this._context) {
    _contextFinalizer.attach(this, _context, detach: this);
  }

  final ffi.Pointer<MpFaceMeshContext> _context;
  bool _closed = false;

  /// Creates the native interpreter and loads a model.
  static Future<MediapipeFaceMesh> create({
    String? modelAssetPath,
    String? modelFilePath,
    String? tfliteLibraryPath,
    int threads = 2,
    double minDetectionConfidence = 0.5,
    double minTrackingConfidence = 0.5,
    bool enableSmoothing = true,
  }) async {
    final String resolvedModelPath =
        modelFilePath ?? await _materializeModel(modelAssetPath);

    final optionsPtr = pkg_ffi.calloc<MpFaceMeshCreateOptions>();
    final ffi.Pointer<pkg_ffi.Utf8> modelPathPtr =
        resolvedModelPath.toNativeUtf8();
    ffi.Pointer<pkg_ffi.Utf8>? libPathPtr;
    try {
      optionsPtr.ref
        ..threads = threads
        ..min_detection_confidence = minDetectionConfidence
        ..min_tracking_confidence = minTrackingConfidence
        ..enable_smoothing = enableSmoothing ? 1 : 0;
      if (tfliteLibraryPath != null && tfliteLibraryPath.isNotEmpty) {
        libPathPtr = tfliteLibraryPath.toNativeUtf8();
        optionsPtr.ref.tflite_library_path = libPathPtr.cast();
      } else {
        optionsPtr.ref.tflite_library_path = ffi.nullptr;
      }

      final ffi.Pointer<MpFaceMeshContext> context =
          faceBindings.mp_face_mesh_create(modelPathPtr.cast(), optionsPtr);
      if (context == ffi.nullptr) {
        throw MediapipeFaceMeshException(
            _readCString(faceBindings.mp_face_mesh_last_global_error()) ??
                'Failed to create face mesh context.');
      }
      return MediapipeFaceMesh._(context);
    } finally {
      pkg_ffi.calloc.free(optionsPtr);
      pkg_ffi.malloc.free(modelPathPtr);
      if (libPathPtr != null) {
        pkg_ffi.malloc.free(libPathPtr);
      }
    }
  }

  static Future<String> _materializeModel(String? assetPath) async {
    final String key = assetPath ?? _defaultModelAsset;
    final ByteData data = await rootBundle.load(key);
    final Directory cacheDir =
        Directory('${Directory.systemTemp.path}/mediapipe_face_mesh_cache');
    if (!await cacheDir.exists()) {
      await cacheDir.create(recursive: true);
    }
    final String sanitizedName = key.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '_');
    final File file = File('${cacheDir.path}/$sanitizedName');
    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    if (!await file.exists() || await file.length() != bytes.length) {
      await file.writeAsBytes(
        bytes,
        flush: true,
      );
    }
    return file.path;
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
    double boxScale = 1.2,
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
    final int logicalWidth =
        (rotationDegrees == 90 || rotationDegrees == 270)
            ? image.height
            : image.width;
    final int logicalHeight =
        (rotationDegrees == 90 || rotationDegrees == 270)
            ? image.width
            : image.height;
    final NormalizedRect? effectiveRoi = roi ??
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
    final ffi.Pointer<MpNormalizedRect> roiPtr =
        effectiveRoi != null ? _toNativeRect(effectiveRoi) : ffi.nullptr;
    FaceMeshResult? processed;
    try {
      final ffi.Pointer<MpFaceMeshResult> resultPtr =
          faceBindings.mp_face_mesh_process(
              _context,
              nativeImage.image,
              roiPtr == ffi.nullptr ? ffi.nullptr : roiPtr,
              rotationDegrees,
              mirrorHorizontal ? 1 : 0,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
            _readCString(faceBindings.mp_face_mesh_last_error(_context)) ??
                'Native face mesh error.');
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
    return processed!;
  }

  /// mirrorHorizontal:If the ROI box is mirrored, mirrorHorizontal should be set to true.
  FaceMeshResult processNv21(
    FaceMeshNv21Image image, {
    NormalizedRect? roi,
    FaceMeshBox? box,
    double boxScale = 1.0,
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
    final int logicalWidth =
        (rotationDegrees == 90 || rotationDegrees == 270)
            ? image.height
            : image.width;
    final int logicalHeight =
        (rotationDegrees == 90 || rotationDegrees == 270)
            ? image.width
            : image.height;
    final NormalizedRect? effectiveRoi = roi ??
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
    final ffi.Pointer<MpNormalizedRect> roiPtr =
        effectiveRoi != null ? _toNativeRect(effectiveRoi) : ffi.nullptr;
    FaceMeshResult? processed;
    try {
      final ffi.Pointer<MpFaceMeshResult> resultPtr =
          faceBindings.mp_face_mesh_process_nv21(
        _context,
        nativeImage.image,
        roiPtr == ffi.nullptr ? ffi.nullptr : roiPtr,
        rotationDegrees,
        mirrorHorizontal ? 1 : 0,
      );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
            _readCString(faceBindings.mp_face_mesh_last_error(_context)) ??
                'Native face mesh error.');
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
    return processed!;
  }

  FaceMeshResult _copyResult(MpFaceMeshResult nativeResult) {
    final ffi.Pointer<MpLandmark> landmarkPtr = nativeResult.landmarks;
    final List<FaceMeshLandmark> landmarks =
        (landmarkPtr == ffi.nullptr || nativeResult.landmarks_count <= 0)
            ? <FaceMeshLandmark>[]
            : List<FaceMeshLandmark>.generate(
                nativeResult.landmarks_count,
                (int i) {
                  final MpLandmark lm = landmarkPtr.elementAt(i).ref;
                  return FaceMeshLandmark(x: lm.x, y: lm.y, z: lm.z);
                },
              );

    return FaceMeshResult(
      landmarks: landmarks,
      rect: NormalizedRect.fromNative(nativeResult.rect),
      score: nativeResult.score,
      imageWidth: nativeResult.image_width,
      imageHeight: nativeResult.image_height,
    );
  }

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

NormalizedRect _normalizedRectFromBox(
  FaceMeshBox box, {
  required int imageWidth,
  required int imageHeight,
  double scale = 1.0,
  bool makeSquare = true,
}) {
  if (imageWidth <= 0 || imageHeight <= 0) {
    throw ArgumentError('Invalid image size: ${imageWidth}x$imageHeight');
  }
  if (!(scale > 0)) {
    throw ArgumentError('scale must be > 0.');
  }
  if (!(box.right > box.left) || !(box.bottom > box.top)) {
    throw ArgumentError('Invalid box: left/top must be < right/bottom.');
  }

  final double clampedLeft =
      box.left.clamp(0.0, imageWidth.toDouble()).toDouble();
  final double clampedTop =
      box.top.clamp(0.0, imageHeight.toDouble()).toDouble();
  final double clampedRight =
      box.right.clamp(0.0, imageWidth.toDouble()).toDouble();
  final double clampedBottom =
      box.bottom.clamp(0.0, imageHeight.toDouble()).toDouble();

  final double centerX = (clampedLeft + clampedRight) * 0.5;
  final double centerY = (clampedTop + clampedBottom) * 0.5;

  double width = (clampedRight - clampedLeft).abs();
  double height = (clampedBottom - clampedTop).abs();

  if (makeSquare) {
    final double size = width > height ? width : height;
    width = size;
    height = size;
  }

  width *= scale;
  height *= scale;

  return NormalizedRect(
    xCenter: centerX / imageWidth,
    yCenter: centerY / imageHeight,
    width: width / imageWidth,
    height: height / imageHeight,
    rotation: 0,
  );
}
