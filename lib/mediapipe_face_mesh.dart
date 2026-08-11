import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart' as pkg_ffi;
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/src/mediapipe_face_bindings_generated.dart';
import 'src/native_bindings_loader.dart';
import 'src/one_euro_filter.dart';

export 'src/one_euro_filter.dart';

part 'src/native_converters.dart';

part 'src/face_mesh_utils.dart';

part 'src/face_mesh_result_utils.dart';

part 'src/face_mesh_geometry.dart';

part 'src/face_mesh_inference_pipeline.dart';

part 'src/face_mesh_landmark_smoothing.dart';

part 'src/face_mesh_topology.dart';

const String _defaultModelAsset =
    'packages/mediapipe_face_mesh/assets/models/mediapipe_face_mesh.tflite';
const String _defaultDetectorModelAsset =
    'packages/mediapipe_face_mesh/assets/models/face_detection_short_range.tflite';
const String _fullRangeDetectorModelAsset =
    'packages/mediapipe_face_mesh/assets/models/face_detection_full_range.tflite';
const String _fullRangeSparseDetectorModelAsset =
    'packages/mediapipe_face_mesh/assets/models/face_detection_full_range_sparse.tflite';
const String _defaultIrisModelAsset =
    'packages/mediapipe_face_mesh/assets/models/iris_landmark.tflite';
const String _attentionModelAsset =
    'packages/mediapipe_face_mesh/assets/models/face_landmark_with_attention.tflite';
const String _defaultBlendshapesModelAsset =
    'packages/mediapipe_face_mesh/assets/models/face_blendshapes.tflite';

/// Bundled MediaPipe face detector model variants.
enum FaceDetectionModel {
  /// Short-range BlazeFace model, best for faces within roughly 2 meters.
  shortRange,

  /// Full-range dense BlazeFace model, best for faces within roughly 5 meters.
  fullRange,

  /// Full-range sparse BlazeFace model optimized for CPU/XNNPACK speed.
  ///
  /// This is the full-range model variant used by the official MediaPipe
  /// Face Detection solution.
  fullRangeSparse,
}

extension on FaceDetectionModel {
  String get assetKey {
    switch (this) {
      case FaceDetectionModel.shortRange:
        return _defaultDetectorModelAsset;
      case FaceDetectionModel.fullRange:
        return _fullRangeDetectorModelAsset;
      case FaceDetectionModel.fullRangeSparse:
        return _fullRangeSparseDetectorModelAsset;
    }
  }

  double get defaultMinDetectionConfidence {
    switch (this) {
      case FaceDetectionModel.shortRange:
        return 0.5;
      case FaceDetectionModel.fullRange:
      case FaceDetectionModel.fullRangeSparse:
        return 0.6;
    }
  }
}

/// Face Mesh landmark indices whose eye coordinates are refined beyond the base
/// mesh: by the separate iris model when
/// `FaceMeshProcessor.create(enableIris: true)` is used, or by the attention
/// model when `enableAttentionMesh: true` is used.
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

/// Default TFLite thread count: half the available cores clamped to 1..4,
/// matching MediaPipe's own CPU inference default.
int _defaultInferenceThreads() {
  final int half = Platform.numberOfProcessors ~/ 2;
  if (half < 1) {
    return 1;
  }
  return half > 4 ? 4 : half;
}

FaceMeshDelegate _faceMeshDelegateFromNative(MpDelegateType delegate) {
  switch (delegate) {
    case MpDelegateType.MP_DELEGATE_CPU:
      return FaceMeshDelegate.cpu;
    case MpDelegateType.MP_DELEGATE_XNNPACK:
      return FaceMeshDelegate.xnnpack;
    case MpDelegateType.MP_DELEGATE_GPU_V2:
      return FaceMeshDelegate.gpuV2;
  }
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

/// Raw image plane data with stride metadata.
///
/// This is useful when adapting camera plugin buffers without depending on a
/// specific camera package type.
class FaceMeshImagePlane {
  /// Creates a raw image plane wrapper.
  const FaceMeshImagePlane({
    required this.bytes,
    required this.bytesPerRow,
    this.bytesPerPixel,
  });

  /// Raw plane bytes.
  final Uint8List bytes;

  /// Bytes consumed per row.
  final int bytesPerRow;

  /// Bytes between adjacent pixels in the same row.
  final int? bytesPerPixel;
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

  /// Converts one contiguous Y + VU plane into a [FaceMeshNv21Image].
  ///
  /// Returns null when dimensions are invalid or [bytes] is too small for the
  /// supplied stride.
  static FaceMeshNv21Image? tryFromSinglePlane({
    required Uint8List bytes,
    required int width,
    required int height,
    required int bytesPerRow,
  }) {
    if (!_isValidNv21Size(width, height) || bytesPerRow <= 0) {
      return null;
    }
    final int ySize = bytesPerRow * height;
    final int vuSize = bytesPerRow * (height ~/ 2);
    if (bytes.length < ySize + vuSize) {
      return null;
    }
    return FaceMeshNv21Image(
      yPlane: Uint8List.sublistView(bytes, 0, ySize),
      vuPlane: Uint8List.sublistView(bytes, ySize, ySize + vuSize),
      width: width,
      height: height,
      yBytesPerRow: bytesPerRow,
      vuBytesPerRow: bytesPerRow,
    );
  }

  /// Converts separate Y and interleaved VU planes into a [FaceMeshNv21Image].
  ///
  /// Plane strides are normalized to tightly-packed output buffers.
  static FaceMeshNv21Image? tryFromYAndInterleavedVuPlanes({
    required int width,
    required int height,
    required FaceMeshImagePlane yPlane,
    required FaceMeshImagePlane vuPlane,
  }) {
    if (!_isValidNv21Size(width, height)) {
      return null;
    }
    final Uint8List? y = _copyPlane(yPlane, width: width, height: height);
    final Uint8List? vu = _copyPlane(
      vuPlane,
      width: width,
      height: height ~/ 2,
    );
    if (y == null || vu == null) {
      return null;
    }
    return FaceMeshNv21Image(
      yPlane: y,
      vuPlane: vu,
      width: width,
      height: height,
      yBytesPerRow: width,
      vuBytesPerRow: width,
    );
  }

  /// Converts YUV420 Y, U, and V planes into a [FaceMeshNv21Image].
  ///
  /// The output chroma plane is converted to MediaPipe's expected interleaved
  /// VU order.
  static FaceMeshNv21Image? tryFromYuv420Planes({
    required int width,
    required int height,
    required FaceMeshImagePlane yPlane,
    required FaceMeshImagePlane uPlane,
    required FaceMeshImagePlane vPlane,
  }) {
    if (!_isValidNv21Size(width, height)) {
      return null;
    }
    final Uint8List? y = _copyPlane(yPlane, width: width, height: height);
    if (y == null) {
      return null;
    }

    final int uvWidth = width ~/ 2;
    final int uvHeight = height ~/ 2;
    final Uint8List vu = Uint8List(width * uvHeight);
    for (var row = 0; row < uvHeight; row++) {
      for (var col = 0; col < uvWidth; col++) {
        final int? u = _readPlaneByte(uPlane, row, col);
        final int? v = _readPlaneByte(vPlane, row, col);
        if (u == null || v == null) {
          return null;
        }
        final int out = row * width + col * 2;
        vu[out] = v;
        vu[out + 1] = u;
      }
    }

    return FaceMeshNv21Image(
      yPlane: y,
      vuPlane: vu,
      width: width,
      height: height,
      yBytesPerRow: width,
      vuBytesPerRow: width,
    );
  }

  static bool _isValidNv21Size(int width, int height) =>
      width > 0 && height > 0 && (width & 1) == 0 && (height & 1) == 0;

  static Uint8List? _copyPlane(
    FaceMeshImagePlane plane, {
    required int width,
    required int height,
  }) {
    if (width <= 0 || height <= 0) {
      return null;
    }
    final Uint8List out = Uint8List(width * height);
    for (var row = 0; row < height; row++) {
      for (var col = 0; col < width; col++) {
        final int? value = _readPlaneByte(plane, row, col);
        if (value == null) {
          return null;
        }
        out[row * width + col] = value;
      }
    }
    return out;
  }

  static int? _readPlaneByte(FaceMeshImagePlane plane, int row, int col) {
    final int pixelStride = plane.bytesPerPixel ?? 1;
    final int index = row * plane.bytesPerRow + col * pixelStride;
    if (pixelStride <= 0 ||
        plane.bytesPerRow <= 0 ||
        index < 0 ||
        index >= plane.bytes.length) {
      return null;
    }
    return plane.bytes[index];
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

/// The 52 ARKit-style face blendshape categories predicted by the MediaPipe
/// face blendshapes model.
///
/// The declaration order matches the model output order, so `.index` maps
/// directly onto the raw coefficient array. [neutral] corresponds to the
/// model's `_neutral` category.
enum FaceBlendshape {
  /// Rest pose; the model's `_neutral` category.
  neutral,

  /// Inner-to-outer lowering of the left brow.
  browDownLeft,

  /// Inner-to-outer lowering of the right brow.
  browDownRight,

  /// Raising of the inner brows.
  browInnerUp,

  /// Raising of the outer left brow.
  browOuterUpLeft,

  /// Raising of the outer right brow.
  browOuterUpRight,

  /// Puffing out of both cheeks.
  cheekPuff,

  /// Upward squint of the left cheek (raising below the eye).
  cheekSquintLeft,

  /// Upward squint of the right cheek (raising below the eye).
  cheekSquintRight,

  /// Closing of the left eyelid.
  eyeBlinkLeft,

  /// Closing of the right eyelid.
  eyeBlinkRight,

  /// Downward gaze of the left eye.
  eyeLookDownLeft,

  /// Downward gaze of the right eye.
  eyeLookDownRight,

  /// Inward (toward the nose) gaze of the left eye.
  eyeLookInLeft,

  /// Inward (toward the nose) gaze of the right eye.
  eyeLookInRight,

  /// Outward (away from the nose) gaze of the left eye.
  eyeLookOutLeft,

  /// Outward (away from the nose) gaze of the right eye.
  eyeLookOutRight,

  /// Upward gaze of the left eye.
  eyeLookUpLeft,

  /// Upward gaze of the right eye.
  eyeLookUpRight,

  /// Narrowing squint of the left eye.
  eyeSquintLeft,

  /// Narrowing squint of the right eye.
  eyeSquintRight,

  /// Widening of the left eye.
  eyeWideLeft,

  /// Widening of the right eye.
  eyeWideRight,

  /// Forward jut of the jaw.
  jawForward,

  /// Leftward movement of the jaw.
  jawLeft,

  /// Opening of the jaw.
  jawOpen,

  /// Rightward movement of the jaw.
  jawRight,

  /// Closing of the lips (independent of jaw).
  mouthClose,

  /// Left dimple.
  mouthDimpleLeft,

  /// Right dimple.
  mouthDimpleRight,

  /// Downward pull of the left mouth corner (frown).
  mouthFrownLeft,

  /// Downward pull of the right mouth corner (frown).
  mouthFrownRight,

  /// Funneling of both lips (an "oh" shape).
  mouthFunnel,

  /// Leftward movement of the mouth.
  mouthLeft,

  /// Lowering of the lower-left lip.
  mouthLowerDownLeft,

  /// Lowering of the lower-right lip.
  mouthLowerDownRight,

  /// Pressing together of the left lips.
  mouthPressLeft,

  /// Pressing together of the right lips.
  mouthPressRight,

  /// Puckering of both lips (a kiss shape).
  mouthPucker,

  /// Rightward movement of the mouth.
  mouthRight,

  /// Rolling of the lower lip inward.
  mouthRollLower,

  /// Rolling of the upper lip inward.
  mouthRollUpper,

  /// Upward shrug of the lower lip.
  mouthShrugLower,

  /// Upward shrug of the upper lip.
  mouthShrugUpper,

  /// Upward pull of the left mouth corner (smile).
  mouthSmileLeft,

  /// Upward pull of the right mouth corner (smile).
  mouthSmileRight,

  /// Sideways stretch of the left mouth corner.
  mouthStretchLeft,

  /// Sideways stretch of the right mouth corner.
  mouthStretchRight,

  /// Raising of the upper-left lip.
  mouthUpperUpLeft,

  /// Raising of the upper-right lip.
  mouthUpperUpRight,

  /// Sneer that raises the left side of the nose.
  noseSneerLeft,

  /// Sneer that raises the right side of the nose.
  noseSneerRight,
}

/// A single 3D landmark returned by MediaPipe.
class FaceMeshLandmark {
  /// Builds a landmark from normalized coordinates returned by MediaPipe.
  FaceMeshLandmark({required this.x, required this.y, required this.z});

  /// Horizontal coordinate normalized to the range `0..1`.
  final double x;

  /// Vertical coordinate normalized to the range `0..1`.
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
  }) : _triangles = triangles;

  /// All face landmarks returned by the native graph.
  final List<FaceMeshLandmark> landmarks;

  List<MpFaceMeshTriangle>? _triangles;

  /// Triangles describing the mesh topology.
  ///
  /// Built lazily on first access so results that are never drawn skip the
  /// 852-triangle construction entirely.
  List<MpFaceMeshTriangle> get triangles =>
      _triangles ??= _buildTrianglesFromLandmarks(landmarks);

  /// Normalized rectangle covering the detected face.
  final NormalizedRect rect;

  /// Confidence score reported by MediaPipe.
  final double score;

  /// Width of the image used during inference.
  final int imageWidth;

  /// Height of the image used during inference.
  final int imageHeight;

  /// Computes the ROI that landmark tracking would use for the next frame.
  ///
  /// This is the landmark bounding box expanded 1.5x into a pixel-space
  /// square, rotated along the eye line (landmarks 33 and 263), with the size
  /// clamped while preserving the aspect ratio. It mirrors the native
  /// `RectFromLandmarks` + `SanitizeRect` implementation in
  /// `src/mediapipe_face_mesh.cc` — keep the two in sync when changing
  /// either.
  ///
  /// Returns a full-frame rect when the result has no usable landmarks.
  NormalizedRect trackingRoi() {
    const NormalizedRect fullFrame = NormalizedRect(
      xCenter: 0.5,
      yCenter: 0.5,
      width: 1,
      height: 1,
    );
    if (landmarks.isEmpty || imageWidth <= 0 || imageHeight <= 0) {
      return fullFrame;
    }
    double minX = 1;
    double minY = 1;
    double maxX = 0;
    double maxY = 0;
    for (final FaceMeshLandmark landmark in landmarks) {
      minX = math.min(minX, landmark.x);
      minY = math.min(minY, landmark.y);
      maxX = math.max(maxX, landmark.x);
      maxY = math.max(maxY, landmark.y);
    }
    final double widthPx = (maxX - minX) * imageWidth;
    final double heightPx = (maxY - minY) * imageHeight;
    if (widthPx < 1e-1 || heightPx < 1e-1) {
      return fullFrame;
    }
    final double longSidePx = math.max(widthPx, heightPx) * 1.5;

    double rotation = 0;
    if (landmarks.length > 263) {
      final FaceMeshLandmark right = landmarks[33];
      final FaceMeshLandmark left = landmarks[263];
      final double dx = (left.x - right.x) * imageWidth;
      final double dy = (left.y - right.y) * imageHeight;
      if (dx.abs() >= 1e-5 || dy.abs() >= 1e-5) {
        rotation = math.atan2(dy, dx);
      }
    }

    final double width = longSidePx / imageWidth;
    final double height = longSidePx / imageHeight;
    // Clamp the size with one scale factor so the width:height ratio
    // (pixel-space squareness) survives clamping.
    final double longDim = math.max(width, height);
    final double shortDim = math.min(width, height);
    double scale = 1.0;
    if (longDim > 2.0) {
      scale = 2.0 / longDim;
    }
    if (shortDim * scale < 0.1) {
      scale = 0.1 / shortDim;
    }
    return NormalizedRect(
      xCenter: ((minX + maxX) * 0.5).clamp(0.0, 1.0),
      yCenter: ((minY + maxY) * 0.5).clamp(0.0, 1.0),
      width: width * scale,
      height: height * scale,
      rotation: rotation,
    );
  }

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
    required double defaultRoiScaleX,
    required double defaultRoiScaleY,
    required double defaultRoiShiftX,
    required double defaultRoiShiftY,
  }) : _defaultRoiScaleX = defaultRoiScaleX,
       _defaultRoiScaleY = defaultRoiScaleY,
       _defaultRoiShiftX = defaultRoiShiftX,
       _defaultRoiShiftY = defaultRoiShiftY {
    _detectorContextFinalizer.attach(this, _context, detach: this);
    _frameScratchFinalizer.attach(this, _scratch, detach: this);
  }

  final ffi.Pointer<MpFaceDetectorContext> _context;
  final double _defaultRoiScaleX;
  final double _defaultRoiScaleY;
  final double _defaultRoiShiftX;
  final double _defaultRoiShiftY;
  final _FrameScratch _scratch = _FrameScratch();
  bool _closed = false;

  /// Delegate that the native detector is actively using after fallback.
  FaceMeshDelegate get activeDelegate {
    _ensureNotClosed();
    return _faceMeshDelegateFromNative(
      faceBindings.mp_face_detector_active_delegate(_context),
    );
  }

  /// Creates the native face detector and loads one of the bundled models.
  ///
  /// Commonly adjusted options:
  /// - [model] selects short-range, full-range dense, or full-range sparse.
  /// - [delegate] selects CPU, XNNPACK, or GPU execution.
  /// - [threads] sets the TFLite thread count. Defaults to half the CPU
  ///   cores clamped to 1..4 (MediaPipe's default).
  /// - [allowDelegateFallback] allows CPU fallback when the requested delegate
  ///   is unavailable, cannot be created, or fails while the interpreter is
  ///   built. Set it to false to fail creation instead.
  /// - [maxResults] limits the number of detections returned per frame.
  /// - [roiScaleX], [roiScaleY], [roiShiftX], and [roiShiftY] control how
  ///   the detector-generated [FaceDetection.expandedFaceRect] is produced.
  ///   The detected face rect is first made square, then scaled by
  ///   [roiScaleX] and [roiScaleY], and shifted by [roiShiftX] and [roiShiftY]
  ///   relative to the original rect size and rotation. Defaults are
  ///   `roiScaleX = 1.5`, `roiScaleY = 1.5`, `roiShiftX = 0.0`, and
  ///   `roiShiftY = 0.0`; the scale defaults match MediaPipe Face Mesh's
  ///   detection-to-ROI transform (`scale_x = 1.5`, `scale_y = 1.5`,
  ///   `square_long = true`).
  static Future<FaceDetectorProcessor> create({
    FaceDetectionModel model = FaceDetectionModel.shortRange,
    int? threads,
    double? minDetectionConfidence,
    double minSuppressionThreshold = 0.3,
    int maxResults = 1,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    bool allowDelegateFallback = true,
    double roiScaleX = 1.5,
    double roiScaleY = 1.5,
    double roiShiftX = 0.0,
    double roiShiftY = 0.0,
  }) async {
    final String resolvedModelPath = await _materializeDetectorModel(model);
    final double resolvedMinDetectionConfidence =
        minDetectionConfidence ?? model.defaultMinDetectionConfidence;

    final optionsPtr = pkg_ffi.calloc<MpFaceDetectorCreateOptions>();
    final ffi.Pointer<pkg_ffi.Utf8> modelPathPtr = resolvedModelPath
        .toNativeUtf8();
    try {
      optionsPtr.ref
        ..threads = threads ?? _defaultInferenceThreads()
        ..min_detection_confidence = resolvedMinDetectionConfidence
        ..min_suppression_threshold = minSuppressionThreshold
        ..max_results = maxResults
        ..delegate = delegate.index
        ..disable_delegate_fallback = allowDelegateFallback ? 0 : 1
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
    final double resolvedRoiScaleX = roiScaleX ?? _defaultRoiScaleX;
    final double resolvedRoiScaleY = roiScaleY ?? _defaultRoiScaleY;
    final double resolvedRoiShiftX = roiShiftX ?? _defaultRoiShiftX;
    final double resolvedRoiShiftY = roiShiftY ?? _defaultRoiShiftY;
    final ffi.Pointer<MpImage> nativeImage = _scratch.imageFrom(image);
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
            nativeImage,
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
      try {
        processed = _copyResult(resultPtr.ref);
      } finally {
        faceBindings.mp_face_detector_release_result(resultPtr);
      }
    } finally {
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
    final double resolvedRoiScaleX = roiScaleX ?? _defaultRoiScaleX;
    final double resolvedRoiScaleY = roiScaleY ?? _defaultRoiScaleY;
    final double resolvedRoiShiftX = roiShiftX ?? _defaultRoiShiftX;
    final double resolvedRoiShiftY = roiShiftY ?? _defaultRoiShiftY;
    final ffi.Pointer<MpNv21Image> nativeImage = _scratch.nv21From(image);
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
            nativeImage,
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
      try {
        processed = _copyResult(resultPtr.ref);
      } finally {
        faceBindings.mp_face_detector_release_result(resultPtr);
      }
    } finally {
      if (roiPtr != ffi.nullptr) pkg_ffi.calloc.free(roiPtr);
      if (roiTransformPtr != ffi.nullptr) pkg_ffi.calloc.free(roiTransformPtr);
    }
    return processed;
  }

  ffi.Pointer<MpRoiTransformOptions> _toNativeRoiTransform(
    double scaleX,
    double scaleY,
    double shiftX,
    double shiftY,
  ) {
    final ffi.Pointer<MpRoiTransformOptions> ptr = pkg_ffi
        .calloc<MpRoiTransformOptions>();
    ptr.ref.scale_x = scaleX;
    ptr.ref.scale_y = scaleY;
    ptr.ref.shift_x = shiftX;
    ptr.ref.shift_y = shiftY;
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
    _frameScratchFinalizer.detach(this);
    _scratch.dispose();
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
  FaceMeshProcessor._(
    this._context, {
    required bool irisEnabled,
    required bool attentionMeshEnabled,
    required bool roiTrackingEnabled,
    required double minTrackingConfidence,
    required double minFacePresenceConfidence,
  }) : _irisEnabled = irisEnabled,
       _attentionMeshEnabled = attentionMeshEnabled,
       _roiTrackingEnabled = roiTrackingEnabled,
       _minTrackingConfidence = minTrackingConfidence,
       _minFacePresenceConfidence = minFacePresenceConfidence {
    _contextFinalizer.attach(this, _context, detach: this);
    _frameScratchFinalizer.attach(this, _scratch, detach: this);
  }

  static const double _boxScale = 1.2;

  final ffi.Pointer<MpFaceMeshContext> _context;
  final bool _irisEnabled;
  final bool _attentionMeshEnabled;
  final bool _roiTrackingEnabled;
  final double _minTrackingConfidence;
  final double _minFacePresenceConfidence;
  final _FrameScratch _scratch = _FrameScratch();
  bool _closed = false;

  /// Whether this processor returns iris landmarks (478 landmarks instead of
  /// the base 468).
  ///
  /// True when created with `enableIris: true` or `enableAttentionMesh: true`.
  bool get irisEnabled => _irisEnabled;

  /// Whether this processor was created with the attention mesh model.
  bool get attentionMeshEnabled => _attentionMeshEnabled;

  /// Whether this processor was created with internal ROI tracking enabled.
  bool get roiTrackingEnabled => _roiTrackingEnabled;

  /// Tracking-confidence threshold this processor was created with.
  ///
  /// [FaceMeshInferencePipeline]'s multi-face flow drops a tracked face when
  /// its mesh presence score falls below this value.
  double get minTrackingConfidence => _minTrackingConfidence;

  /// Face-presence threshold this processor was created with.
  ///
  /// Frames whose mesh presence score falls below this value return a
  /// [FaceMeshResult] with no landmarks.
  double get minFacePresenceConfidence => _minFacePresenceConfidence;

  /// Whether the internal tracked ROI is currently following a face.
  ///
  /// True after a mesh call ([process]/[processNv21], or their
  /// [processRois]/[processNv21Rois] batch forms) seeded the ROI from face
  /// landmarks; false initially, after a face-presence or
  /// tracking-confidence failure dropped the ROI, or after an input
  /// rotation/mirroring change reset it. Always false when this processor
  /// was created with `enableRoiTracking: false`.
  bool get isTracking {
    _ensureNotClosed();
    return faceBindings.mp_face_mesh_is_tracking(_context) != 0;
  }

  /// Delegate that the native face mesh model is actively using after fallback.
  FaceMeshDelegate get activeDelegate {
    _ensureNotClosed();
    return _faceMeshDelegateFromNative(
      faceBindings.mp_face_mesh_active_delegate(_context),
    );
  }

  /// Delegate that the separate iris model is actively using after fallback.
  ///
  /// Returns null when no separate iris pass runs — either because this
  /// processor was created with `enableIris: false`, or because
  /// `enableAttentionMesh: true` produces the iris landmarks inside the mesh
  /// inference ([activeDelegate] is the delegate running it).
  FaceMeshDelegate? get activeIrisDelegate {
    _ensureNotClosed();
    if (!_irisEnabled || _attentionMeshEnabled) {
      return null;
    }
    return _faceMeshDelegateFromNative(
      faceBindings.mp_face_mesh_active_iris_delegate(_context),
    );
  }

  /// Creates the native interpreter and loads a model.
  ///
  /// Commonly adjusted options:
  /// - [delegate] selects CPU, XNNPACK, or GPU execution.
  /// - [threads] sets the TFLite thread count. Defaults to half the CPU
  ///   cores clamped to 1..4 (MediaPipe's default).
  /// - [allowDelegateFallback] allows CPU fallback when the requested delegate
  ///   is unavailable, cannot be created, or fails while the interpreter is
  ///   built. Set it to false to fail creation instead.
  /// - [enableSmoothing] smooths the internally tracked ROI across frames
  ///   (used when [roi]/[box] are omitted), which stabilizes the crop fed to
  ///   the model and indirectly reduces landmark jitter. It does not filter
  ///   the landmark coordinates themselves.
  /// - [enableRoiTracking] reuses internal ROI tracking when [roi] or [box]
  ///   are omitted in later [process] or [processNv21] calls.
  /// - [minTrackingConfidence] is the mesh presence score below which
  ///   tracking stops trusting a followed face. The multi-face pipeline flow
  ///   drops the track and re-acquires it via the detector. The single-face
  ///   native tracking drops its internal ROI the same way (observable
  ///   through [isTracking]); [FaceMeshInferencePipeline] re-acquires via
  ///   the detector, while raw [process] calls without an ROI fall back to
  ///   full-frame inference on the next frame.
  /// - [minFacePresenceConfidence] is the mesh presence score below which a
  ///   frame is treated as having no usable face: the result carries no
  ///   landmarks, and calls without an explicit ROI also reset internal ROI
  ///   tracking. Scores are compared after sigmoid, like the official
  ///   graph's face-presence threshold.
  /// - [enableIris] runs a separate iris pass after the base 468-point mesh: it
  ///   refines the eye landmarks and appends iris landmarks, returning 478
  ///   landmarks instead of the base 468 landmarks.
  /// - [enableAttentionMesh] replaces the base mesh model with the unified
  ///   `face_landmark_with_attention` model, which refines lips, eyes, and
  ///   irises in a single inference and returns the same 478-landmark layout
  ///   with better accuracy around those regions. Iris is always included, so
  ///   it supersedes [enableIris]: when both are set the separate iris model is
  ///   not loaded and [activeIrisDelegate] is null, while [irisEnabled] stays
  ///   true and iris-dependent consumers such as [FaceBlendshapesProcessor]
  ///   keep working.
  static Future<FaceMeshProcessor> create({
    int? threads,
    double minDetectionConfidence = 0.5,
    double minTrackingConfidence = 0.5,
    double minFacePresenceConfidence = 0.5,
    bool enableSmoothing = true,
    bool enableRoiTracking = true,
    bool enableIris = false,
    bool enableAttentionMesh = false,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    bool allowDelegateFallback = true,
  }) async {
    // The attention model already includes refined irises in its 478 output, so
    // it replaces the base mesh model and the separate iris pass.
    final bool irisIncluded = enableIris || enableAttentionMesh;
    final String resolvedModelPath = enableAttentionMesh
        ? await _materializeAttentionModel()
        : await _materializeModel();
    final String? resolvedIrisModelPath =
        (enableIris && !enableAttentionMesh)
            ? await _materializeIrisModel()
            : null;

    final optionsPtr = pkg_ffi.calloc<MpFaceMeshCreateOptions>();
    final ffi.Pointer<pkg_ffi.Utf8> modelPathPtr = resolvedModelPath
        .toNativeUtf8();
    final ffi.Pointer<pkg_ffi.Utf8> irisModelPathPtr =
        resolvedIrisModelPath?.toNativeUtf8() ?? ffi.nullptr;
    try {
      optionsPtr.ref
        ..threads = threads ?? _defaultInferenceThreads()
        ..min_detection_confidence = minDetectionConfidence
        ..min_tracking_confidence = minTrackingConfidence
        ..min_face_presence_confidence = minFacePresenceConfidence
        ..delegate = delegate.index
        ..disable_delegate_fallback = allowDelegateFallback ? 0 : 1
        ..enable_smoothing = enableSmoothing ? 1 : 0
        ..enable_roi_tracking = enableRoiTracking ? 1 : 0
        ..enable_iris = enableIris ? 1 : 0
        ..enable_attention_mesh = enableAttentionMesh ? 1 : 0
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
      return FaceMeshProcessor._(
        context,
        irisEnabled: irisIncluded,
        attentionMeshEnabled: enableAttentionMesh,
        roiTrackingEnabled: enableRoiTracking,
        minTrackingConfidence: minTrackingConfidence,
        minFacePresenceConfidence: minFacePresenceConfidence,
      );
    } finally {
      pkg_ffi.calloc.free(optionsPtr);
      pkg_ffi.malloc.free(modelPathPtr);
      if (irisModelPathPtr != ffi.nullptr) {
        pkg_ffi.malloc.free(irisModelPathPtr);
      }
    }
  }

  /// Creates a face mesh processor configured for multi-face ROI fan-out.
  ///
  /// Multi-face helpers run several face ROIs through the same processor in
  /// sequence. Smoothing and ROI tracking keep state across calls, so this
  /// factory disables both options to prevent state from one face affecting the
  /// next face.
  static Future<FaceMeshProcessor> createForMultiFace({
    int? threads,
    double minDetectionConfidence = 0.5,
    double minTrackingConfidence = 0.5,
    double minFacePresenceConfidence = 0.5,
    bool enableIris = false,
    bool enableAttentionMesh = false,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    bool allowDelegateFallback = true,
  }) {
    return FaceMeshProcessor.create(
      threads: threads,
      minDetectionConfidence: minDetectionConfidence,
      minTrackingConfidence: minTrackingConfidence,
      minFacePresenceConfidence: minFacePresenceConfidence,
      enableSmoothing: false,
      enableRoiTracking: false,
      enableIris: enableIris,
      enableAttentionMesh: enableAttentionMesh,
      delegate: delegate,
      allowDelegateFallback: allowDelegateFallback,
    );
  }

  /// Processes an image and returns face landmarks.
  ///
  /// By default, this processes using the internal ROI tracking state when
  /// [enableRoiTracking] was enabled during [create].
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
    _validateRotationDegrees(rotationDegrees);
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
    final ffi.Pointer<MpImage> nativeImage = _scratch.imageFrom(image);
    final ffi.Pointer<MpNormalizedRect> roiPtr = effectiveRoi != null
        ? _toNativeRect(effectiveRoi)
        : ffi.nullptr;
    FaceMeshResult? processed;
    try {
      final ffi.Pointer<MpFaceMeshResult> resultPtr = faceBindings
          .mp_face_mesh_process(
            _context,
            nativeImage,
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
      try {
        processed = _copyResult(resultPtr.ref);
      } finally {
        faceBindings.mp_face_mesh_release_result(resultPtr);
      }
    } finally {
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
    _validateRotationDegrees(rotationDegrees);
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
    final ffi.Pointer<MpNv21Image> nativeImage = _scratch.nv21From(image);
    final ffi.Pointer<MpNormalizedRect> roiPtr = effectiveRoi != null
        ? _toNativeRect(effectiveRoi)
        : ffi.nullptr;
    FaceMeshResult? processed;
    try {
      final ffi.Pointer<MpFaceMeshResult> resultPtr = faceBindings
          .mp_face_mesh_process_nv21(
            _context,
            nativeImage,
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
      try {
        processed = _copyResult(resultPtr.ref);
      } finally {
        faceBindings.mp_face_mesh_release_result(resultPtr);
      }
    } finally {
      if (roiPtr != ffi.nullptr) {
        pkg_ffi.calloc.free(roiPtr);
      }
    }
    return processed;
  }

  /// Runs one mesh inference per ROI on a single native frame upload.
  ///
  /// Unlike calling [process] once per face, the frame is copied into native
  /// memory once regardless of how many ROIs are provided. Returns one
  /// [FaceMeshResult] per ROI, in input order; results whose face presence
  /// score fell below the threshold have no landmarks, matching [process].
  ///
  /// Like [process] with an explicit ROI, each successful inference seeds
  /// the internal tracked ROI when the processor was created with
  /// `enableRoiTracking: true` — after this call it follows the last entry
  /// in [rois] that produced landmarks (observable through [isTracking]).
  List<FaceMeshResult> processRois(
    FaceMeshImage image, {
    required List<NormalizedRect> rois,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    _ensureNotClosed();
    _validateRotationDegrees(rotationDegrees);
    if (rois.isEmpty) {
      return <FaceMeshResult>[];
    }
    final ffi.Pointer<MpImage> nativeImage = _scratch.imageFrom(image);
    final ffi.Pointer<MpNormalizedRect> roisPtr = _toNativeRectArray(rois);
    try {
      final ffi.Pointer<MpFaceMeshMultiResult> resultPtr = faceBindings
          .mp_face_mesh_process_rois(
            _context,
            nativeImage,
            roisPtr,
            rois.length,
            rotationDegrees,
            mirrorHorizontal ? 1 : 0,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_mesh_last_error(_context)) ??
              'Native face mesh error.',
        );
      }
      try {
        return _copyMultiResult(resultPtr.ref);
      } finally {
        faceBindings.mp_face_mesh_release_multi_result(resultPtr);
      }
    } finally {
      pkg_ffi.calloc.free(roisPtr);
    }
  }

  /// Runs one NV21 mesh inference per ROI on a single native frame upload.
  ///
  /// This is the NV21 counterpart of [processRois]; see that method for the
  /// result semantics.
  List<FaceMeshResult> processNv21Rois(
    FaceMeshNv21Image image, {
    required List<NormalizedRect> rois,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    _ensureNotClosed();
    _validateRotationDegrees(rotationDegrees);
    if (rois.isEmpty) {
      return <FaceMeshResult>[];
    }
    final ffi.Pointer<MpNv21Image> nativeImage = _scratch.nv21From(image);
    final ffi.Pointer<MpNormalizedRect> roisPtr = _toNativeRectArray(rois);
    try {
      final ffi.Pointer<MpFaceMeshMultiResult> resultPtr = faceBindings
          .mp_face_mesh_process_rois_nv21(
            _context,
            nativeImage,
            roisPtr,
            rois.length,
            rotationDegrees,
            mirrorHorizontal ? 1 : 0,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_face_mesh_last_error(_context)) ??
              'Native face mesh error.',
        );
      }
      try {
        return _copyMultiResult(resultPtr.ref);
      } finally {
        faceBindings.mp_face_mesh_release_multi_result(resultPtr);
      }
    } finally {
      pkg_ffi.calloc.free(roisPtr);
    }
  }

  /// Processes one mesh inference for each detector result with a usable ROI.
  ///
  /// This mirrors MediaPipe Face Mesh graph behavior at the Dart API level:
  /// each [FaceDetection.expandedFaceRect], or [FaceDetection.faceRect] when
  /// the expanded ROI is unavailable, is run through one mesh inference and
  /// collected into a single list. The frame is uploaded to native memory
  /// once for all faces (see [processRois]).
  ///
  /// [maxMeshFaces] limits how many mesh inferences are run from the provided
  /// [detections]. Detections without an ROI are skipped.
  List<FaceMeshResult> processMultiFace(
    FaceMeshImage image, {
    required Iterable<FaceDetection> detections,
    int? maxMeshFaces,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    _validateMaxMeshFaces(maxMeshFaces);
    return processRois(
      image,
      rois: _roisForDetections(detections, maxMeshFaces),
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
    );
  }

  /// Processes one NV21 mesh inference for each detector result with a usable
  /// ROI.
  ///
  /// This is the NV21 counterpart of [processMultiFace].
  ///
  /// [maxMeshFaces] limits how many mesh inferences are run from the provided
  /// [detections]. Detections without an ROI are skipped.
  List<FaceMeshResult> processNv21MultiFace(
    FaceMeshNv21Image image, {
    required Iterable<FaceDetection> detections,
    int? maxMeshFaces,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    _validateMaxMeshFaces(maxMeshFaces);
    return processNv21Rois(
      image,
      rois: _roisForDetections(detections, maxMeshFaces),
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
    );
  }

  List<NormalizedRect> _roisForDetections(
    Iterable<FaceDetection> detections,
    int? maxMeshFaces,
  ) {
    final List<NormalizedRect> rois = <NormalizedRect>[];
    for (final FaceDetection detection in detections) {
      if (maxMeshFaces != null && rois.length >= maxMeshFaces) {
        break;
      }
      final NormalizedRect? roi = _roiForDetection(detection);
      if (roi == null) {
        continue;
      }
      rois.add(roi);
    }
    return rois;
  }

  FaceMeshResult _copyResult(MpFaceMeshResult nativeResult) {
    final ffi.Pointer<MpLandmark> landmarkPtr = nativeResult.landmarks;
    final int landmarkCount = nativeResult.landmarks_count;
    final List<FaceMeshLandmark> landmarks;
    if (landmarkPtr == ffi.nullptr || landmarkCount <= 0) {
      landmarks = <FaceMeshLandmark>[];
    } else {
      // MpLandmark is three packed floats, so one typed-data view over the
      // whole array avoids materializing a struct view per landmark.
      final Float32List xyz = landmarkPtr.cast<ffi.Float>().asTypedList(
        landmarkCount * 3,
      );
      landmarks = List<FaceMeshLandmark>.generate(landmarkCount, (int i) {
        final int base = i * 3;
        return FaceMeshLandmark(
          x: xyz[base],
          y: xyz[base + 1],
          z: xyz[base + 2],
        );
      }, growable: false);
    }

    return FaceMeshResult(
      landmarks: landmarks,
      rect: NormalizedRect.fromNative(nativeResult.rect),
      score: nativeResult.score,
      imageWidth: nativeResult.image_width,
      imageHeight: nativeResult.image_height,
    );
  }

  List<FaceMeshResult> _copyMultiResult(MpFaceMeshMultiResult nativeResult) {
    final ffi.Pointer<MpFaceMeshResult> resultsPtr = nativeResult.results;
    if (resultsPtr == ffi.nullptr || nativeResult.results_count <= 0) {
      return <FaceMeshResult>[];
    }
    return List<FaceMeshResult>.generate(
      nativeResult.results_count,
      (int i) => _copyResult((resultsPtr + i).ref),
    );
  }

  /// Releases the native context and associated resources.
  void close() {
    if (_closed) {
      return;
    }
    _contextFinalizer.detach(this);
    _frameScratchFinalizer.detach(this);
    _scratch.dispose();
    faceBindings.mp_face_mesh_destroy(_context);
    _closed = true;
  }

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Face mesh context already closed.');
    }
  }

  void _validateRotationDegrees(int rotationDegrees) {
    if (rotationDegrees != 0 &&
        rotationDegrees != 90 &&
        rotationDegrees != 180 &&
        rotationDegrees != 270) {
      throw ArgumentError('rotationDegrees must be one of {0, 90, 180, 270}.');
    }
  }

  NormalizedRect? _roiForDetection(FaceDetection detection) =>
      detection.expandedFaceRect ?? detection.faceRect;

  void _validateMaxMeshFaces(int? maxMeshFaces) {
    if (maxMeshFaces != null && maxMeshFaces < 0) {
      throw ArgumentError('maxMeshFaces must be null or >= 0.');
    }
  }
}

final Finalizer<ffi.Pointer<MpBlendshapesContext>>
_blendshapesContextFinalizer =
    Finalizer<ffi.Pointer<MpBlendshapesContext>>(
      (pointer) => faceBindings.mp_blendshapes_destroy(pointer),
    );

/// A post-processor that turns face landmarks into 52 ARKit-style blendshape
/// coefficients.
///
/// This is a separate processor from [FaceMeshProcessor]: create one once, then
/// call [process] on any [FaceMeshResult] whose landmarks include the iris
/// points (i.e. produced by a mesh processor created with `enableIris: true` or
/// `enableAttentionMesh: true`).
class FaceBlendshapesProcessor {
  FaceBlendshapesProcessor._(this._context) {
    _blendshapesContextFinalizer.attach(this, _context, detach: this);
  }

  /// Number of landmarks the blendshapes model requires (468 mesh + 10 iris).
  static const int requiredLandmarkCount = 478;

  final ffi.Pointer<MpBlendshapesContext> _context;
  bool _closed = false;

  /// Delegate the blendshapes model is actively using after fallback.
  FaceMeshDelegate get activeDelegate {
    _ensureNotClosed();
    return _faceMeshDelegateFromNative(
      faceBindings.mp_blendshapes_active_delegate(_context),
    );
  }

  /// Loads the bundled face blendshapes model.
  ///
  /// - [delegate] selects CPU, XNNPACK, or GPU execution.
  /// - [threads] sets the TFLite thread count. Defaults to half the CPU
  ///   cores clamped to 1..4 (MediaPipe's default).
  /// - [allowDelegateFallback] allows CPU fallback when the requested delegate
  ///   is unavailable, cannot be created, or fails while the interpreter is
  ///   built. Set it to false to fail creation instead.
  static Future<FaceBlendshapesProcessor> create({
    int? threads,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    bool allowDelegateFallback = true,
  }) async {
    final String resolvedModelPath = await _materializeBlendshapesModel();
    final optionsPtr = pkg_ffi.calloc<MpBlendshapesCreateOptions>();
    final ffi.Pointer<pkg_ffi.Utf8> modelPathPtr = resolvedModelPath
        .toNativeUtf8();
    try {
      optionsPtr.ref
        ..threads = threads ?? _defaultInferenceThreads()
        ..delegate = delegate.index
        ..disable_delegate_fallback = allowDelegateFallback ? 0 : 1
        ..tflite_library_path = ffi.nullptr;

      final ffi.Pointer<MpBlendshapesContext> context = faceBindings
          .mp_blendshapes_create(modelPathPtr.cast(), optionsPtr);
      if (context == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_blendshapes_last_global_error()) ??
              'Failed to create blendshapes context.',
        );
      }
      return FaceBlendshapesProcessor._(context);
    } finally {
      pkg_ffi.calloc.free(optionsPtr);
      pkg_ffi.malloc.free(modelPathPtr);
    }
  }

  /// Runs the blendshapes model on [result]'s landmarks and returns the 52
  /// coefficients keyed by category.
  ///
  /// Returns null when [result] has no landmarks (no face was present in the
  /// frame).
  ///
  /// Throws [ArgumentError] when [result] carries some landmarks but fewer than
  /// [requiredLandmarkCount] — this means the source mesh was created without
  /// `enableIris: true` or `enableAttentionMesh: true`, one of which the
  /// blendshapes model requires for the iris landmarks.
  Map<FaceBlendshape, double>? process(FaceMeshResult result) {
    _ensureNotClosed();
    final List<FaceMeshLandmark> landmarks = result.landmarks;
    if (landmarks.isEmpty) {
      return null;
    }
    if (landmarks.length < requiredLandmarkCount) {
      throw ArgumentError(
        'FaceBlendshapesProcessor requires $requiredLandmarkCount landmarks '
        '(create the FaceMeshProcessor with enableIris: true or '
        'enableAttentionMesh: true); got ${landmarks.length}.',
      );
    }

    final ffi.Pointer<MpLandmark> landmarkPtr = pkg_ffi.calloc<MpLandmark>(
      landmarks.length,
    );
    try {
      for (var i = 0; i < landmarks.length; i++) {
        final FaceMeshLandmark landmark = landmarks[i];
        landmarkPtr[i]
          ..x = landmark.x
          ..y = landmark.y
          ..z = landmark.z;
      }
      final ffi.Pointer<MpBlendshapesResult> resultPtr = faceBindings
          .mp_blendshapes_process(
            _context,
            landmarkPtr,
            landmarks.length,
            result.imageWidth,
            result.imageHeight,
          );
      if (resultPtr == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          _readCString(faceBindings.mp_blendshapes_last_error(_context)) ??
              'Native blendshapes error.',
        );
      }
      try {
        return _copyScores(resultPtr.ref);
      } finally {
        faceBindings.mp_blendshapes_release_result(resultPtr);
      }
    } finally {
      pkg_ffi.calloc.free(landmarkPtr);
    }
  }

  Map<FaceBlendshape, double> _copyScores(MpBlendshapesResult nativeResult) {
    final ffi.Pointer<ffi.Float> ptr = nativeResult.scores;
    final int count = nativeResult.scores_count;
    if (ptr == ffi.nullptr || count < FaceBlendshape.values.length) {
      throw MediapipeFaceMeshException(
        'Unexpected blendshapes output size: $count.',
      );
    }
    return <FaceBlendshape, double>{
      for (final FaceBlendshape shape in FaceBlendshape.values)
        shape: (ptr + shape.index).value,
    };
  }

  /// Releases the native blendshapes context and associated resources.
  void close() {
    if (_closed) {
      return;
    }
    _blendshapesContextFinalizer.detach(this);
    faceBindings.mp_blendshapes_destroy(_context);
    _closed = true;
  }

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Blendshapes context already closed.');
    }
  }
}
