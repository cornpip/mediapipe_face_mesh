import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;

import 'package:ffi/ffi.dart' as pkg_ffi;
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/src/mediapipe_face_bindings_generated.dart';
import 'src/native_bindings_loader.dart';

part 'src/native_converters.dart';

part 'src/face_mesh_utils.dart';

part 'src/face_mesh_result_utils.dart';

part 'src/face_mesh_topology.dart';

const String _defaultModelAsset =
    'packages/mediapipe_face_mesh/assets/models/mediapipe_face_mesh.tflite';
const String _defaultDetectorModelAsset =
    'packages/mediapipe_face_mesh/assets/models/face_detection_short_range.tflite';
const String _defaultIrisModelAsset =
    'packages/mediapipe_face_mesh/assets/models/iris_landmark.tflite';

/// Face Mesh landmark indices whose coordinates are refined by the iris model
/// when `FaceMeshProcessor.create(enableIris: true)` is used.
const Set<int> faceMeshIrisRefinedEyeLandmarkIndices = <int>{
  33,
  7,
  163,
  144,
  145,
  153,
  154,
  155,
  133,
  246,
  161,
  160,
  159,
  158,
  157,
  173,
  130,
  25,
  110,
  24,
  23,
  22,
  26,
  112,
  243,
  247,
  30,
  29,
  27,
  28,
  56,
  190,
  226,
  31,
  228,
  229,
  230,
  231,
  232,
  233,
  244,
  113,
  225,
  224,
  223,
  222,
  221,
  189,
  35,
  124,
  46,
  53,
  52,
  65,
  143,
  111,
  117,
  118,
  119,
  120,
  121,
  128,
  245,
  156,
  70,
  63,
  105,
  66,
  107,
  55,
  193,
  263,
  249,
  390,
  373,
  374,
  380,
  381,
  382,
  362,
  466,
  388,
  387,
  386,
  385,
  384,
  398,
  359,
  255,
  339,
  254,
  253,
  252,
  256,
  341,
  463,
  467,
  260,
  259,
  257,
  258,
  286,
  414,
  446,
  261,
  448,
  449,
  450,
  451,
  452,
  453,
  464,
  342,
  445,
  444,
  443,
  442,
  441,
  413,
  265,
  353,
  276,
  283,
  282,
  295,
  372,
  340,
  346,
  347,
  348,
  349,
  350,
  357,
  465,
  383,
  300,
  293,
  334,
  296,
  336,
  285,
  417,
};

final Finalizer<ffi.Pointer<MpFaceMeshContext>> _contextFinalizer =
    Finalizer<ffi.Pointer<MpFaceMeshContext>>(
      (pointer) => faceBindings.mp_face_mesh_destroy(pointer),
    );
final Finalizer<ffi.Pointer<MpFaceDetectorContext>> _detectorContextFinalizer =
    Finalizer<ffi.Pointer<MpFaceDetectorContext>>(
      (pointer) => faceBindings.mp_face_detector_destroy(pointer),
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

  /// Returns a new [NormalizedRect] with scale, shift, and optional squaring
  /// applied — mirroring the C++ `RectTransformationCalculator` logic.
  ///
  /// All shifts and scales operate in the **face's own coordinate system**
  /// (i.e. rotated by [rotation]), so the result is correct even for tilted
  /// faces.
  ///
  /// Parameters:
  /// - [scaleX] / [scaleY]: multiply width / height (default 1.0, no change).
  /// - [squareLong]: if true, both sides are set to `max(width, height)` before
  ///   scaling (equivalent to MediaPipe's `square_long: true`).
  /// - [shiftX] / [shiftY]: shift the center in the face's local axes,
  ///   expressed as a fraction of the **original** (pre-scale) width / height.
  ///   Negative [shiftY] moves toward the top of the head.
  NormalizedRect transform({
    double scaleX = 1.0,
    double scaleY = 1.0,
    bool squareLong = false,
    double shiftX = 0.0,
    double shiftY = 0.0,
  }) {
    double outWidth = width;
    double outHeight = height;
    if (squareLong) {
      final longSide = outWidth > outHeight ? outWidth : outHeight;
      outWidth = longSide;
      outHeight = longSide;
    }

    final cosR = math.cos(rotation);
    final sinR = math.sin(rotation);

    // Shift uses the original (pre-squareLong) dimensions, matching C++.
    final newCenterX = xCenter + width * shiftX * cosR - height * shiftY * sinR;
    final newCenterY = yCenter + width * shiftX * sinR + height * shiftY * cosR;

    return NormalizedRect(
      xCenter: newCenterX,
      yCenter: newCenterY,
      width: outWidth * scaleX,
      height: outHeight * scaleY,
      rotation: rotation,
    );
  }

  @override
  String toString() =>
      'NormalizedRect(xCenter: $xCenter, yCenter: $yCenter, width: $width, '
      'height: $height, rotation: $rotation)';
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

  @override
  String toString() =>
      'FaceMeshBox(left: $left, top: $top, right: $right, bottom: $bottom)';
}

/// Single face detection in normalized image coordinates.
class FaceDetection {
  /// Creates a normalized face detection box.
  const FaceDetection({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.score,
    this.faceRect,
    this.expandedFaceRect,
  });

  /// Left edge in normalized coordinates.
  final double left;

  /// Top edge in normalized coordinates.
  final double top;

  /// Right edge in normalized coordinates.
  final double right;

  /// Bottom edge in normalized coordinates.
  final double bottom;

  /// Confidence score reported by the detector.
  final double score;

  /// Rotation-aware rect derived from the detection keypoints.
  final NormalizedRect? faceRect;

  /// Expanded face ROI that matches MediaPipe's rect transformation step.
  final NormalizedRect? expandedFaceRect;

  /// Converts this normalized detection into a pixel-space [FaceMeshBox].
  FaceMeshBox toBox({required int imageWidth, required int imageHeight}) =>
      FaceMeshBox(
        left: left * imageWidth,
        top: top * imageHeight,
        right: right * imageWidth,
        bottom: bottom * imageHeight,
      );

  /// Convenience normalized ROI for directly feeding Face Mesh.
  NormalizedRect toNormalizedRect({
    double scale = 1.0,
    bool makeSquare = false,
  }) {
    if (!(scale > 0)) {
      throw ArgumentError('scale must be > 0.');
    }
    double width = (right - left).abs();
    double height = (bottom - top).abs();
    if (makeSquare) {
      final double size = width > height ? width : height;
      width = size;
      height = size;
    }
    width *= scale;
    height *= scale;
    return NormalizedRect(
      xCenter: (left + right) * 0.5,
      yCenter: (top + bottom) * 0.5,
      width: width,
      height: height,
    );
  }

  @override
  String toString() =>
      'FaceDetection(left: $left, top: $top, right: $right, bottom: $bottom, '
      'score: $score, faceRect: $faceRect, expandedFaceRect: $expandedFaceRect)';
}

/// Result of a face detector inference.
class FaceDetectionResult {
  /// Creates a detection result container.
  const FaceDetectionResult({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// All detections sorted by descending score.
  final List<FaceDetection> detections;

  /// Width of the image used during inference.
  final int imageWidth;

  /// Height of the image used during inference.
  final int imageHeight;

  /// Highest-confidence detection when present.
  FaceDetection? get primaryDetection =>
      detections.isEmpty ? null : detections.first;

  @override
  String toString() =>
      'FaceDetectionResult(detections: ${detections.length}, imageWidth: '
      '$imageWidth, imageHeight: $imageHeight)';
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

  @override
  String toString() =>
      'FaceMeshImage(width: $width, height: $height, bytesPerRow: $bytesPerRow, '
      'pixelFormat: $pixelFormat, pixelsLength: ${pixels.length})';
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

  @override
  String toString() =>
      'FaceMeshNv21Image(width: $width, height: $height, '
      'yBytesPerRow: $yBytesPerRow, vuBytesPerRow: $vuBytesPerRow, '
      'yPlaneLength: ${yPlane.length}, vuPlaneLength: ${vuPlane.length})';
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

  @override
  String toString() => 'FaceMeshLandmark(x: $x, y: $y, z: $z)';
}

/// Triangle made up of 3 face mesh landmarks.
class MpFaceMeshTriangle {
  /// Builds a triangle from landmark indices and the referenced points.
  MpFaceMeshTriangle({required this.indices, required this.points});

  /// Indices into the full landmark list (length 3).
  final List<int> indices;

  /// Landmark points referenced by [indices] (length 3).
  final List<FaceMeshLandmark> points;

  @override
  String toString() => 'MpFaceMeshTriangle(indices: $indices)';
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
    List<MpFaceMeshTriangle>? triangles,
  }) : triangles = triangles ?? _buildTrianglesFromLandmarks(landmarks);

  /// All face landmarks returned by the native graph.
  final List<FaceMeshLandmark> landmarks;

  /// Triangles describing the mesh topology.
  final List<MpFaceMeshTriangle> triangles;

  /// Normalized rectangle covering the detected face.
  final NormalizedRect rect;

  /// Confidence score reported by MediaPipe.
  final double score;

  /// Width of the image used during inference.
  final int imageWidth;

  /// Height of the image used during inference.
  final int imageHeight;

  @override
  String toString() =>
      'FaceMeshResult(landmarks: ${landmarks.length}, triangles: '
      '${triangles.length}, rect: $rect, score: $score, imageWidth: '
      '$imageWidth, imageHeight: $imageHeight)';
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

/// High-level wrapper around the native MediaPipe Face Detection model.
class FaceDetectorProcessor {
  FaceDetectorProcessor._(
    this._context, {
    required double? defaultRoiScaleX,
    required double? defaultRoiScaleY,
    required double defaultRoiShiftX,
    required double defaultRoiShiftY,
  }) : _defaultRoiScaleX = defaultRoiScaleX,
       _defaultRoiScaleY = defaultRoiScaleY,
       _defaultRoiShiftX = defaultRoiShiftX,
       _defaultRoiShiftY = defaultRoiShiftY {
    _detectorContextFinalizer.attach(this, _context, detach: this);
  }

  final ffi.Pointer<MpFaceDetectorContext> _context;
  final double? _defaultRoiScaleX;
  final double? _defaultRoiScaleY;
  final double _defaultRoiShiftX;
  final double _defaultRoiShiftY;
  bool _closed = false;

  /// Creates the native face detector and loads the bundled short-range model.
  ///
  /// Commonly adjusted options:
  /// - [delegate] selects CPU, XNNPACK, or GPU execution.
  /// - [maxResults] limits the number of detections returned per frame.
  /// - [roiScaleX], [roiScaleY], [roiShiftX], and [roiShiftY] control how
  ///   the detector-generated [FaceDetection.expandedFaceRect] is produced.
  static Future<FaceDetectorProcessor> create({
    int threads = 2,
    double minDetectionConfidence = 0.5,
    double minSuppressionThreshold = 0.3,
    int maxResults = 1,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    double? roiScaleX,
    double? roiScaleY,
    double roiShiftX = 0.0,
    double roiShiftY = 0.0,
  }) async {
    final String resolvedModelPath = await _materializeDetectorModel();

    final optionsPtr = pkg_ffi.calloc<MpFaceDetectorCreateOptions>();
    final ffi.Pointer<pkg_ffi.Utf8> modelPathPtr = resolvedModelPath
        .toNativeUtf8();
    try {
      optionsPtr.ref
        ..threads = threads
        ..min_detection_confidence = minDetectionConfidence
        ..min_suppression_threshold = minSuppressionThreshold
        ..max_results = maxResults
        ..delegate = delegate.index
        ..tflite_library_path = ffi.nullptr;

      final ffi.Pointer<MpFaceDetectorContext> context = faceBindings
          .mp_face_detector_create(modelPathPtr.cast(), optionsPtr);
      if (context == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_detector_last_global_error()) ??
              'Failed to create face detector context.',
        );
      }
      return FaceDetectorProcessor._(
        context,
        defaultRoiScaleX: roiScaleX,
        defaultRoiScaleY: roiScaleY,
        defaultRoiShiftX: roiShiftX,
        defaultRoiShiftY: roiShiftY,
      );
    } finally {
      pkg_ffi.calloc.free(optionsPtr);
      pkg_ffi.malloc.free(modelPathPtr);
    }
  }

  /// Processes an RGBA/BGRA frame and returns normalized face detections.
  FaceDetectionResult process(
    FaceMeshImage image, {
    NormalizedRect? roi,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? roiScaleX,
    double? roiScaleY,
    double? roiShiftX,
    double? roiShiftY,
  }) {
    _ensureNotClosed();
    _validateRotation(rotationDegrees);
    final double? resolvedRoiScaleX = roiScaleX ?? _defaultRoiScaleX;
    final double? resolvedRoiScaleY = roiScaleY ?? _defaultRoiScaleY;
    final double resolvedRoiShiftX = roiShiftX ?? _defaultRoiShiftX;
    final double resolvedRoiShiftY = roiShiftY ?? _defaultRoiShiftY;
    final _NativeImage nativeImage = _toNativeImage(image);
    final ffi.Pointer<MpNormalizedRect> roiPtr = roi != null
        ? _toNativeRect(roi)
        : ffi.nullptr;
    final ffi.Pointer<MpRoiTransformOptions> roiTransformPtr =
        _toNativeRoiTransform(
          resolvedRoiScaleX,
          resolvedRoiScaleY,
          resolvedRoiShiftX,
          resolvedRoiShiftY,
        );
    FaceDetectionResult? processed;
    try {
      final ffi.Pointer<MpFaceDetectorResult> resultPtr = faceBindings
          .mp_face_detector_process(
            _context,
            nativeImage.image,
            roiPtr == ffi.nullptr ? ffi.nullptr : roiPtr,
            rotationDegrees,
            mirrorHorizontal ? 1 : 0,
            roiTransformPtr,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_detector_last_error(_context)) ??
              'Native face detector error.',
        );
      }
      processed = _copyResult(resultPtr.ref);
      faceBindings.mp_face_detector_release_result(resultPtr);
    } finally {
      pkg_ffi.calloc.free(nativeImage.pixels);
      pkg_ffi.calloc.free(nativeImage.image);
      if (roiPtr != ffi.nullptr) pkg_ffi.calloc.free(roiPtr);
      if (roiTransformPtr != ffi.nullptr) pkg_ffi.calloc.free(roiTransformPtr);
    }
    return processed;
  }

  /// Processes an NV21 frame and returns normalized face detections.
  FaceDetectionResult processNv21(
    FaceMeshNv21Image image, {
    NormalizedRect? roi,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? roiScaleX,
    double? roiScaleY,
    double? roiShiftX,
    double? roiShiftY,
  }) {
    _ensureNotClosed();
    _validateRotation(rotationDegrees);
    final double? resolvedRoiScaleX = roiScaleX ?? _defaultRoiScaleX;
    final double? resolvedRoiScaleY = roiScaleY ?? _defaultRoiScaleY;
    final double resolvedRoiShiftX = roiShiftX ?? _defaultRoiShiftX;
    final double resolvedRoiShiftY = roiShiftY ?? _defaultRoiShiftY;
    final _NativeNv21Image nativeImage = _toNativeNv21Image(image);
    final ffi.Pointer<MpNormalizedRect> roiPtr = roi != null
        ? _toNativeRect(roi)
        : ffi.nullptr;
    final ffi.Pointer<MpRoiTransformOptions> roiTransformPtr =
        _toNativeRoiTransform(
          resolvedRoiScaleX,
          resolvedRoiScaleY,
          resolvedRoiShiftX,
          resolvedRoiShiftY,
        );
    FaceDetectionResult? processed;
    try {
      final ffi.Pointer<MpFaceDetectorResult> resultPtr = faceBindings
          .mp_face_detector_process_nv21(
            _context,
            nativeImage.image,
            roiPtr == ffi.nullptr ? ffi.nullptr : roiPtr,
            rotationDegrees,
            mirrorHorizontal ? 1 : 0,
            roiTransformPtr,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_detector_last_error(_context)) ??
              'Native face detector error.',
        );
      }
      processed = _copyResult(resultPtr.ref);
      faceBindings.mp_face_detector_release_result(resultPtr);
    } finally {
      pkg_ffi.calloc.free(nativeImage.yPlane);
      pkg_ffi.calloc.free(nativeImage.vuPlane);
      pkg_ffi.calloc.free(nativeImage.image);
      if (roiPtr != ffi.nullptr) pkg_ffi.calloc.free(roiPtr);
      if (roiTransformPtr != ffi.nullptr) pkg_ffi.calloc.free(roiTransformPtr);
    }
    return processed;
  }

  ffi.Pointer<MpRoiTransformOptions> _toNativeRoiTransform(
    double? scaleX,
    double? scaleY,
    double? shiftX,
    double? shiftY,
  ) {
    if (scaleX == null && scaleY == null && shiftX == null && shiftY == null) {
      return ffi.nullptr;
    }
    final ffi.Pointer<MpRoiTransformOptions> ptr = pkg_ffi
        .calloc<MpRoiTransformOptions>();
    ptr.ref.scale_x = (scaleX ?? 1.5).toDouble();
    ptr.ref.scale_y = (scaleY ?? 1.5).toDouble();
    ptr.ref.shift_x = shiftX ?? 0.0;
    ptr.ref.shift_y = shiftY ?? 0.0;
    return ptr;
  }

  FaceDetectionResult _copyResult(MpFaceDetectorResult nativeResult) {
    final ffi.Pointer<MpDetection> detectionPtr = nativeResult.detections;
    final List<FaceDetection> detections =
        (detectionPtr == ffi.nullptr || nativeResult.detections_count <= 0)
        ? <FaceDetection>[]
        : List<FaceDetection>.generate(nativeResult.detections_count, (int i) {
            final MpDetection detection = (detectionPtr + i).ref;
            return FaceDetection(
              left: detection.left,
              top: detection.top,
              right: detection.right,
              bottom: detection.bottom,
              score: detection.score,
              faceRect: NormalizedRect.fromNative(detection.face_rect),
              expandedFaceRect: NormalizedRect.fromNative(
                detection.expanded_face_rect,
              ),
            );
          });

    return FaceDetectionResult(
      detections: detections,
      imageWidth: nativeResult.image_width,
      imageHeight: nativeResult.image_height,
    );
  }

  /// Releases the native detector context and associated resources.
  void close() {
    if (_closed) {
      return;
    }
    _detectorContextFinalizer.detach(this);
    faceBindings.mp_face_detector_destroy(_context);
    _closed = true;
  }

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Face detector context already closed.');
    }
  }

  void _validateRotation(int rotationDegrees) {
    if (rotationDegrees != 0 &&
        rotationDegrees != 90 &&
        rotationDegrees != 180 &&
        rotationDegrees != 270) {
      throw ArgumentError('rotationDegrees must be one of {0, 90, 180, 270}.');
    }
  }
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
  ///
  /// Commonly adjusted options:
  /// - [delegate] selects CPU, XNNPACK, or GPU execution.
  /// - [enableSmoothing] reduces landmark jitter across frames.
  /// - [enableRoiTracking] reuses internal ROI tracking when [roi] or [box]
  ///   are omitted in later [process] or [processNv21] calls.
  /// - [enableIris] refines eye landmarks and appends iris landmarks, returning
  ///   478 landmarks instead of the base 468 landmarks.
  static Future<FaceMeshProcessor> create({
    int threads = 2,
    double minDetectionConfidence = 0.5,
    double minTrackingConfidence = 0.5,
    bool enableSmoothing = true,
    bool enableRoiTracking = true,
    bool enableIris = false,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
  }) async {
    final String resolvedModelPath = await _materializeModel();
    final String? resolvedIrisModelPath = enableIris
        ? await _materializeIrisModel()
        : null;

    final optionsPtr = pkg_ffi.calloc<MpFaceMeshCreateOptions>();
    final ffi.Pointer<pkg_ffi.Utf8> modelPathPtr = resolvedModelPath
        .toNativeUtf8();
    final ffi.Pointer<pkg_ffi.Utf8> irisModelPathPtr =
        resolvedIrisModelPath?.toNativeUtf8() ?? ffi.nullptr;
    try {
      optionsPtr.ref
        ..threads = threads
        ..min_detection_confidence = minDetectionConfidence
        ..min_tracking_confidence = minTrackingConfidence
        ..delegate = delegate.index
        ..enable_smoothing = enableSmoothing ? 1 : 0
        ..enable_roi_tracking = enableRoiTracking ? 1 : 0
        ..enable_iris = enableIris ? 1 : 0
        ..iris_model_path = irisModelPathPtr.cast()
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
      if (irisModelPathPtr != ffi.nullptr) {
        pkg_ffi.malloc.free(irisModelPathPtr);
      }
    }
  }

  /// Processes an image and returns face landmarks.
  ///
  /// By default, this processes using the internal ROI tracking state.
  /// To process the full frame or restrict processing to a region, provide either:
  /// - [roi] as a normalized rectangle, or
  /// - [box] as a pixel-space bounding box (converted to an ROI internally).
  ///
  /// To force full-frame inference without passing a region each time, disable
  /// ROI tracking at creation via [enableRoiTracking].
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
