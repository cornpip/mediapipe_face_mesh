import 'dart:math' as math;

import 'package:flutter/foundation.dart' show listEquals;
import 'package:flutter/material.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// A mesh ROI to draw, with an optional label such as a track id.
class FaceRoi {
  /// Creates an ROI overlay entry.
  const FaceRoi(this.rect, {this.label});

  /// The rotated ROI in normalized image coordinates, as reported by
  /// [FaceMeshInferenceResult.selectedRoi], [TrackedFaceMesh.mesh], or
  /// [FaceMeshResult.rect].
  final NormalizedRect rect;

  /// Text drawn above the box. Null draws no label.
  final String? label;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    if (other is! FaceRoi || other.label != label) {
      return false;
    }
    final NormalizedRect a = rect;
    final NormalizedRect b = other.rect;
    return a.xCenter == b.xCenter &&
        a.yCenter == b.yCenter &&
        a.width == b.width &&
        a.height == b.height &&
        a.rotation == b.rotation;
  }

  @override
  int get hashCode => Object.hash(
    rect.xCenter,
    rect.yCenter,
    rect.width,
    rect.height,
    rect.rotation,
    label,
  );
}

/// Draws [FaceDetection] boxes, their mesh ROIs, and [extraRois].
/// [FaceDetectionPainter.fromInference] and
/// [FaceDetectionPainter.fromMultiInference] fill both from a pipeline
/// result.
///
/// This painter is intended for debug and preview overlays. It depends only on
/// this package's detection types, so camera-specific mirroring decisions should
/// be converted to [mirrorHorizontal] by the caller.
class FaceDetectionPainter extends CustomPainter {
  /// Creates a painter from a detector result, extra ROIs, or both.
  FaceDetectionPainter({
    this.result,
    this.extraRois = const <FaceRoi>[],
    this.rotationDegrees = 0,
    this.mirrorHorizontal = false,
    this.showConfidence = true,
    this.showFaceBox = true,
    this.showRoiBox = false,
    this.faceBoxColor = Colors.amberAccent,
    this.roiBoxColor = Colors.lightGreenAccent,
    this.strokeWidth = 2.0,
    this.roiStrokeWidth = 3.0,
    this.bboxLabel = 'Face',
    this.roiLabel = 'ROI',
    this.labelTextStyle = const TextStyle(
      color: Colors.black87,
      fontSize: 14,
      fontWeight: FontWeight.w600,
    ),
  }) {
    _validateRotationDegrees(rotationDegrees);
  }

  /// Draws the detections when the detector ran, plus the ROI the mesh ran
  /// on with [label] above it.
  FaceDetectionPainter.fromInference(
    FaceMeshInferenceResult inference, {
    String? label,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    bool showConfidence = true,
    bool showFaceBox = true,
    bool showRoiBox = false,
    Color faceBoxColor = Colors.amberAccent,
    Color roiBoxColor = Colors.lightGreenAccent,
    double strokeWidth = 2.0,
    double roiStrokeWidth = 3.0,
    String bboxLabel = 'Face',
    String roiLabel = 'ROI',
    TextStyle labelTextStyle = const TextStyle(
      color: Colors.black87,
      fontSize: 14,
      fontWeight: FontWeight.w600,
    ),
  }) : this(
         result: inference.detectionResult,
         extraRois: <FaceRoi>[
           if (inference.selectedRoi != null)
             FaceRoi(inference.selectedRoi!, label: label),
         ],
         rotationDegrees: rotationDegrees,
         mirrorHorizontal: mirrorHorizontal,
         showConfidence: showConfidence,
         showFaceBox: showFaceBox,
         showRoiBox: showRoiBox,
         faceBoxColor: faceBoxColor,
         roiBoxColor: roiBoxColor,
         strokeWidth: strokeWidth,
         roiStrokeWidth: roiStrokeWidth,
         bboxLabel: bboxLabel,
         roiLabel: roiLabel,
         labelTextStyle: labelTextStyle,
       );

  /// Draws the detections, if the detector ran, plus each tracked face's
  /// ROI. [labelOf] names a face's ROI and defaults to `#` followed by its
  /// track id.
  FaceDetectionPainter.fromMultiInference(
    FaceMeshMultiInferenceResult inference, {
    String Function(TrackedFaceMesh face)? labelOf,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    bool showConfidence = true,
    bool showFaceBox = true,
    bool showRoiBox = false,
    Color faceBoxColor = Colors.amberAccent,
    Color roiBoxColor = Colors.lightGreenAccent,
    double strokeWidth = 2.0,
    double roiStrokeWidth = 3.0,
    String bboxLabel = 'Face',
    String roiLabel = 'ROI',
    TextStyle labelTextStyle = const TextStyle(
      color: Colors.black87,
      fontSize: 14,
      fontWeight: FontWeight.w600,
    ),
  }) : this(
         result: inference.detectionResult,
         extraRois: <FaceRoi>[
           for (final TrackedFaceMesh face in inference.faces)
             FaceRoi(
               face.mesh.rect,
               label: labelOf?.call(face) ?? '#${face.trackId}',
             ),
         ],
         rotationDegrees: rotationDegrees,
         mirrorHorizontal: mirrorHorizontal,
         showConfidence: showConfidence,
         showFaceBox: showFaceBox,
         showRoiBox: showRoiBox,
         faceBoxColor: faceBoxColor,
         roiBoxColor: roiBoxColor,
         strokeWidth: strokeWidth,
         roiStrokeWidth: roiStrokeWidth,
         bboxLabel: bboxLabel,
         roiLabel: roiLabel,
         labelTextStyle: labelTextStyle,
       );

  /// Detector result to draw. Null on landmark-tracked frames, where the
  /// pipeline reports no detection.
  final FaceDetectionResult? result;

  /// ROIs drawn in addition to [result], with [roiBoxColor] and
  /// [roiStrokeWidth].
  final List<FaceRoi> extraRois;

  /// Clockwise rotation applied when mapping normalized coordinates to pixels.
  final int rotationDegrees;

  /// Whether X coordinates should be mirrored for preview overlays.
  final bool mirrorHorizontal;

  /// Whether confidence scores are appended to face box labels.
  final bool showConfidence;

  /// Whether axis-aligned detector boxes are drawn.
  final bool showFaceBox;

  /// Whether detector-produced rotated mesh ROIs are drawn.
  final bool showRoiBox;

  /// Color used for axis-aligned detector boxes.
  final Color faceBoxColor;

  /// Color used for rotated ROI boxes.
  final Color roiBoxColor;

  /// Stroke width for axis-aligned detector boxes.
  final double strokeWidth;

  /// Stroke width for rotated ROI boxes.
  final double roiStrokeWidth;

  /// Label shown for axis-aligned detector boxes. Empty draws no label.
  final String bboxLabel;

  /// Label shown for rotated ROI boxes. Empty draws no label.
  final String roiLabel;

  /// Text style used for labels.
  final TextStyle labelTextStyle;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint faceBoxPaint = Paint()
      ..color = faceBoxColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth;
    final Paint roiPaint = Paint()
      ..color = roiBoxColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = roiStrokeWidth;

    for (final FaceRoi roi in extraRois) {
      final Path path = _buildRotatedRectPath(roi.rect, size);
      canvas.drawPath(path, roiPaint);
      _paintLabel(
        canvas,
        anchorRect: _rotatedRectBounds(roi.rect, size),
        label: roi.label ?? '',
        color: roiBoxColor,
      );
    }

    final FaceDetectionResult? result = this.result;
    if (result == null) {
      return;
    }
    for (final FaceDetection detection in result.detections) {
      if (showFaceBox) {
        final Rect rawRect = _mapBox(detection, size);
        canvas.drawRect(rawRect, faceBoxPaint);
        _paintLabel(
          canvas,
          anchorRect: rawRect,
          label: showConfidence
              ? '$bboxLabel ${(detection.score * 100).toStringAsFixed(1)}%'
              : bboxLabel,
          color: faceBoxColor,
        );
      }

      if (showRoiBox) {
        final NormalizedRect roi = detection.expandedFaceRect;
        final Path path = _buildRotatedRectPath(roi, size);
        canvas.drawPath(path, roiPaint);
        _paintLabel(
          canvas,
          anchorRect: _rotatedRectBounds(roi, size),
          label: roiLabel,
          color: roiBoxColor,
        );
      }
    }
  }

  Rect _mapBox(FaceDetection detection, Size size) {
    final List<Offset> corners = <Offset>[
      Offset(detection.left, detection.top),
      Offset(detection.right, detection.top),
      Offset(detection.right, detection.bottom),
      Offset(detection.left, detection.bottom),
    ].map(_mapNormalizedPoint).toList();
    double minX = corners.first.dx;
    double minY = corners.first.dy;
    double maxX = corners.first.dx;
    double maxY = corners.first.dy;
    for (final Offset corner in corners.skip(1)) {
      minX = math.min(minX, corner.dx);
      minY = math.min(minY, corner.dy);
      maxX = math.max(maxX, corner.dx);
      maxY = math.max(maxY, corner.dy);
    }
    return Rect.fromLTRB(
      minX * size.width,
      minY * size.height,
      maxX * size.width,
      maxY * size.height,
    );
  }

  void _paintLabel(
    Canvas canvas, {
    required Rect anchorRect,
    required String label,
    required Color color,
  }) {
    if (label.isEmpty) {
      return;
    }
    final TextPainter textPainter = TextPainter(
      text: TextSpan(text: label, style: labelTextStyle),
      textDirection: TextDirection.ltr,
    )..layout();

    final Rect textBackground = Rect.fromLTWH(
      anchorRect.left,
      math.max(0, anchorRect.top - textPainter.height - 4),
      textPainter.width + 8,
      textPainter.height + 4,
    );

    final Paint backgroundPaint = Paint()
      ..color = color.withValues(alpha: 0.85)
      ..style = PaintingStyle.fill;
    canvas.drawRect(textBackground, backgroundPaint);
    textPainter.paint(
      canvas,
      Offset(textBackground.left + 4, textBackground.top + 2),
    );
  }

  Path _buildRotatedRectPath(NormalizedRect rect, Size size) {
    final List<Offset> corners = _rotatedRectCorners(rect, size);
    return Path()
      ..moveTo(corners[0].dx, corners[0].dy)
      ..lineTo(corners[1].dx, corners[1].dy)
      ..lineTo(corners[2].dx, corners[2].dy)
      ..lineTo(corners[3].dx, corners[3].dy)
      ..close();
  }

  Rect _rotatedRectBounds(NormalizedRect rect, Size size) {
    final List<Offset> corners = _rotatedRectCorners(rect, size);
    double minX = corners.first.dx;
    double minY = corners.first.dy;
    double maxX = corners.first.dx;
    double maxY = corners.first.dy;
    for (final Offset corner in corners.skip(1)) {
      minX = math.min(minX, corner.dx);
      minY = math.min(minY, corner.dy);
      maxX = math.max(maxX, corner.dx);
      maxY = math.max(maxY, corner.dy);
    }
    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  List<Offset> _rotatedRectCorners(NormalizedRect rect, Size size) {
    final Size sourceSize = rotationDegrees == 90 || rotationDegrees == 270
        ? Size(size.height, size.width)
        : size;
    final double centerX = rect.xCenter * sourceSize.width;
    final double centerY = rect.yCenter * sourceSize.height;
    final double width = rect.width * sourceSize.width;
    final double height = rect.height * sourceSize.height;
    final double cosR = math.cos(rect.rotation);
    final double sinR = math.sin(rect.rotation);
    final List<Offset> localCorners = <Offset>[
      Offset(-width * 0.5, -height * 0.5),
      Offset(width * 0.5, -height * 0.5),
      Offset(width * 0.5, height * 0.5),
      Offset(-width * 0.5, height * 0.5),
    ];
    return localCorners.map((Offset corner) {
      final Offset sourcePoint = Offset(
        centerX + cosR * corner.dx - sinR * corner.dy,
        centerY + sinR * corner.dx + cosR * corner.dy,
      );
      return _mapSourcePixelPoint(sourcePoint, sourceSize, size);
    }).toList();
  }

  Offset _mapSourcePixelPoint(Offset point, Size sourceSize, Size targetSize) {
    double x = point.dx;
    double y = point.dy;

    switch (rotationDegrees) {
      case 90:
        x = sourceSize.height - point.dy;
        y = point.dx;
        break;
      case 180:
        x = sourceSize.width - point.dx;
        y = sourceSize.height - point.dy;
        break;
      case 270:
        x = point.dy;
        y = sourceSize.width - point.dx;
        break;
      case 0:
        break;
    }

    if (mirrorHorizontal) {
      x = targetSize.width - x;
    }
    return Offset(x, y);
  }

  Offset _mapNormalizedPoint(Offset point) {
    double x = point.dx;
    double y = point.dy;

    switch (rotationDegrees) {
      case 90:
        x = 1.0 - point.dy;
        y = point.dx;
        break;
      case 180:
        x = 1.0 - point.dx;
        y = 1.0 - point.dy;
        break;
      case 270:
        x = point.dy;
        y = 1.0 - point.dx;
        break;
      case 0:
        break;
    }

    if (mirrorHorizontal) {
      x = 1.0 - x;
    }
    return Offset(x, y);
  }

  void _validateRotationDegrees(int rotationDegrees) {
    if (rotationDegrees != 0 &&
        rotationDegrees != 90 &&
        rotationDegrees != 180 &&
        rotationDegrees != 270) {
      throw ArgumentError('rotationDegrees must be one of {0, 90, 180, 270}.');
    }
  }

  @override
  bool shouldRepaint(covariant FaceDetectionPainter oldDelegate) {
    return oldDelegate.result != result ||
        !listEquals(oldDelegate.extraRois, extraRois) ||
        oldDelegate.rotationDegrees != rotationDegrees ||
        oldDelegate.mirrorHorizontal != mirrorHorizontal ||
        oldDelegate.showConfidence != showConfidence ||
        oldDelegate.showFaceBox != showFaceBox ||
        oldDelegate.showRoiBox != showRoiBox ||
        oldDelegate.faceBoxColor != faceBoxColor ||
        oldDelegate.roiBoxColor != roiBoxColor ||
        oldDelegate.strokeWidth != strokeWidth ||
        oldDelegate.roiStrokeWidth != roiStrokeWidth ||
        oldDelegate.bboxLabel != bboxLabel ||
        oldDelegate.roiLabel != roiLabel ||
        oldDelegate.labelTextStyle != labelTextStyle;
  }
}
