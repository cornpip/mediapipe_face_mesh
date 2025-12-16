import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart' as pkg_ffi;
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_bindings_generated.dart';

import 'mediapipe_face_mesh.dart';
import 'src/bindings.dart';

class FaceDetection {
  FaceDetection({
    required this.rect,
    required this.score,
    required this.keypoints,
  });

  /// Normalized bounding box.
  final NormalizedRect rect;
  final double score;
  /// Normalized keypoints: list of (x,y) pairs in image coordinates.
  final List<Offset> keypoints;
}

class FaceDetectionResult {
  FaceDetectionResult({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  final List<FaceDetection> detections;
  final int imageWidth;
  final int imageHeight;
}

class FaceDetectionOptions {
  const FaceDetectionOptions({
    this.tfliteLibraryPath,
    this.threads = 2,
    this.scoreThreshold = 0.5,
    this.nmsThreshold = 0.3,
    this.maxDetections = 1,
  });

  final String? tfliteLibraryPath;
  final int threads;
  final double scoreThreshold;
  final double nmsThreshold;
  final int maxDetections;
}

class MediapipeFaceDetection {
  MediapipeFaceDetection._(this._context) {
    _fdFinalizer.attach(this, _context, detach: this);
  }

  final ffi.Pointer<MpFaceDetectionContext> _context;
  bool _closed = false;
  static final Finalizer<ffi.Pointer<MpFaceDetectionContext>> _fdFinalizer =
      Finalizer<ffi.Pointer<MpFaceDetectionContext>>(
          (pointer) => faceBindings.mp_face_detection_destroy(pointer));

  static Future<String> _fdMaterializeModel(String assetPath) async {
    final ByteData data = await rootBundle.load(assetPath);
    final Directory cacheDir =
        Directory('${Directory.systemTemp.path}/mediapipe_face_mesh_cache');
    if (!await cacheDir.exists()) {
      await cacheDir.create(recursive: true);
    }
    final String sanitizedName =
        assetPath.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '_');
    final File file = File('${cacheDir.path}/$sanitizedName');
    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    if (!await file.exists() || await file.length() != bytes.length) {
      await file.writeAsBytes(bytes, flush: true);
    }
    return file.path;
  }

  static Future<MediapipeFaceDetection> create({
    String? modelAssetPath,
    String? modelFilePath,
    FaceDetectionOptions options = const FaceDetectionOptions(),
  }) async {
    final String resolvedModelPath =
        modelFilePath ??
        await _fdMaterializeModel(
            modelAssetPath ??
                'packages/mediapipe_face_mesh/assets/models/mediapipe_face_detection_short_range.tflite');

    final ffi.Pointer<MpFaceDetectionCreateOptions> optPtr =
        pkg_ffi.calloc<MpFaceDetectionCreateOptions>();
    ffi.Pointer<pkg_ffi.Utf8>? libPtr;
    try {
      optPtr.ref
        ..threads = options.threads
        ..score_threshold = options.scoreThreshold
        ..nms_threshold = options.nmsThreshold
        ..max_detections = options.maxDetections;
      if (options.tfliteLibraryPath != null &&
          options.tfliteLibraryPath!.isNotEmpty) {
        libPtr = options.tfliteLibraryPath!.toNativeUtf8();
        optPtr.ref.tflite_library_path = libPtr.cast();
      } else {
        optPtr.ref.tflite_library_path = ffi.nullptr;
      }

      final ffi.Pointer<MpFaceDetectionContext> ctx =
          faceBindings.mp_face_detection_create(
              resolvedModelPath.toNativeUtf8().cast(), optPtr);
      if (ctx == ffi.nullptr) {
        throw MediapipeFaceMeshException(
            _fdReadCString(
                faceBindings.mp_face_detection_last_global_error()) ??
                'Failed to create face detection context.');
      }
      return MediapipeFaceDetection._(ctx);
    } finally {
      pkg_ffi.calloc.free(optPtr);
      if (libPtr != null) {
        pkg_ffi.malloc.free(libPtr);
      }
    }
  }

  FaceDetectionResult process(FaceMeshImage image) {
    _ensureNotClosed();
    final _FdNativeImage nativeImage = _fdToNativeImage(image);
    FaceDetectionResult? processed;
    try {
      final ffi.Pointer<MpFaceDetectionResult> resultPtr =
          faceBindings.mp_face_detection_process(_context, nativeImage.image);
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
            _fdReadCString(
                faceBindings.mp_face_detection_last_error(_context)) ??
                'Native face detection error.');
      }
      processed = _copyDetectionResult(resultPtr.ref);
      faceBindings.mp_face_detection_release_result(resultPtr);
    } finally {
      pkg_ffi.calloc.free(nativeImage.pixels);
      pkg_ffi.calloc.free(nativeImage.image);
    }
    return processed!;
  }

  FaceDetectionResult _copyDetectionResult(
      MpFaceDetectionResult nativeResult) {
    final List<FaceDetection> detections = [];
    final int count = nativeResult.count;
    final ffi.Pointer<MpDetection> ptr = nativeResult.detections;
    for (int i = 0; i < count; ++i) {
      final MpDetection d = ptr.elementAt(i).ref;
      final NormalizedRect rect = NormalizedRect(
        xCenter: d.box.x_center,
        yCenter: d.box.y_center,
        width: d.box.width,
        height: d.box.height,
      );
      final List<Offset> kps = [];
      final int kpCount = d.keypoints_count;
      for (int k = 0; k < kpCount; ++k) {
        final double x = d.keypoints[k * 2];
        final double y = d.keypoints[k * 2 + 1];
        kps.add(Offset(x, y));
      }
      detections.add(FaceDetection(
          rect: rect, score: d.score.toDouble(), keypoints: kps));
    }
    return FaceDetectionResult(
      detections: detections,
      imageWidth: nativeResult.image_width,
      imageHeight: nativeResult.image_height,
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _fdFinalizer.detach(this);
    faceBindings.mp_face_detection_destroy(_context);
    _closed = true;
  }

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Face detection context already closed.');
    }
  }
}

class _FdNativeImage {
  _FdNativeImage({
    required this.image,
    required this.pixels,
  });

  final ffi.Pointer<MpImage> image;
  final ffi.Pointer<ffi.Uint8> pixels;
}

_FdNativeImage _fdToNativeImage(FaceMeshImage image) {
  final ffi.Pointer<MpImage> imagePtr = pkg_ffi.calloc<MpImage>();
  final ffi.Pointer<ffi.Uint8> pixelPtr =
      pkg_ffi.calloc<ffi.Uint8>(image.pixels.length);
  pixelPtr.asTypedList(image.pixels.length).setAll(0, image.pixels);
  imagePtr.ref
    ..data = pixelPtr.cast()
    ..width = image.width
    ..height = image.height
    ..bytes_per_row = image.bytesPerRow
    ..format = image.pixelFormat;
  return _FdNativeImage(image: imagePtr, pixels: pixelPtr);
}

String? _fdReadCString(ffi.Pointer<ffi.Char> pointer) {
  if (pointer == ffi.nullptr) {
    return null;
  }
  return pointer.cast<pkg_ffi.Utf8>().toDartString();
}
