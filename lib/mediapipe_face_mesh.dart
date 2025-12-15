import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart' as pkg_ffi;
import 'package:flutter/services.dart';

import 'mediapipe_face_mesh_bindings_generated.dart';

const String _libName = 'mediapipe_face_mesh';
const String _defaultModelAsset =
    'packages/mediapipe_face_mesh/assets/models/mediapipe_face_mesh.tflite';

final ffi.DynamicLibrary _dylib = () {
  if (Platform.isMacOS || Platform.isIOS) {
    return ffi.DynamicLibrary.open('$_libName.framework/$_libName');
  }
  if (Platform.isAndroid || Platform.isLinux) {
    return ffi.DynamicLibrary.open('lib$_libName.so');
  }
  if (Platform.isWindows) {
    return ffi.DynamicLibrary.open('$_libName.dll');
  }
  throw UnsupportedError('Unsupported platform ${Platform.operatingSystem}');
}();

final MediapipeFaceMeshBindings _bindings = MediapipeFaceMeshBindings(_dylib);
final Finalizer<ffi.Pointer<MpFaceMeshContext>> _contextFinalizer =
    Finalizer<ffi.Pointer<MpFaceMeshContext>>(
        (pointer) => _bindings.mp_face_mesh_destroy(pointer));

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
          _bindings.mp_face_mesh_create(modelPathPtr.cast(), optionsPtr);
      if (context == ffi.nullptr) {
        throw MediapipeFaceMeshException(
            _readCString(_bindings.mp_face_mesh_last_global_error()) ??
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

  FaceMeshResult process(FaceMeshImage image, {NormalizedRect? roi}) {
    _ensureNotClosed();
    final imagePtr = pkg_ffi.calloc<MpImage>();
    final roiPtr = roi != null ? pkg_ffi.calloc<MpNormalizedRect>() : ffi.nullptr;
    final pixelPtr = pkg_ffi.calloc<ffi.Uint8>(image.pixels.length);
    FaceMeshResult? processed;
    try {
      pixelPtr.asTypedList(image.pixels.length).setAll(0, image.pixels);
      imagePtr.ref
        ..data = pixelPtr.cast()
        ..width = image.width
        ..height = image.height
        ..bytes_per_row = image.bytesPerRow
        ..format = image.pixelFormat;

      if (roiPtr != ffi.nullptr) {
        roiPtr.ref
          ..x_center = roi!.xCenter
          ..y_center = roi.yCenter
          ..width = roi.width
          ..height = roi.height
          ..rotation = roi.rotation;
      }

      final ffi.Pointer<MpFaceMeshResult> resultPtr =
          _bindings.mp_face_mesh_process(
              _context, imagePtr, roiPtr == ffi.nullptr ? ffi.nullptr : roiPtr);
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
            _readCString(_bindings.mp_face_mesh_last_error(_context)) ??
                'Native face mesh error.');
      }
      processed = _copyResult(resultPtr.ref);
      _bindings.mp_face_mesh_release_result(resultPtr);
    } finally {
      pkg_ffi.calloc.free(pixelPtr);
      pkg_ffi.calloc.free(imagePtr);
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
    _bindings.mp_face_mesh_destroy(_context);
    _closed = true;
  }

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Face mesh context already closed.');
    }
  }
}

String? _readCString(ffi.Pointer<ffi.Char> pointer) {
  if (pointer == ffi.nullptr) {
    return null;
  }
  return pointer.cast<pkg_ffi.Utf8>().toDartString();
}
