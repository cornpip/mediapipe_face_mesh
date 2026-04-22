import 'dart:io' show Platform;
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

class Detection {
  const Detection({
    required this.boundingBox,
    required this.confidence,
    required this.bboxLabel,
    this.roiLabel = 'ROI',
    this.rotatedRect,
  });

  final Rect boundingBox;
  final double confidence;
  final String bboxLabel;
  final String roiLabel;
  final NormalizedRect? rotatedRect;
}

class DetectionPainter extends CustomPainter {
  DetectionPainter({
    required this.detections,
    required this.lensDirection,
    this.showConfidence = true,
  });

  final List<Detection> detections;
  final CameraLensDirection lensDirection;
  final bool showConfidence;

  @override
  void paint(Canvas canvas, Size size) {
    final rawBoxPaint = Paint()
      ..color = Colors.amberAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    final roiPaint = Paint()
      ..color = Colors.lightGreenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (final detection in detections) {
      final mirroredBox = _maybeMirror(detection.boundingBox);
      final rawRect = Rect.fromLTRB(
        mirroredBox.left * size.width,
        mirroredBox.top * size.height,
        mirroredBox.right * size.width,
        mirroredBox.bottom * size.height,
      );
      canvas.drawRect(rawRect, rawBoxPaint);
      _paintLabel(
        canvas,
        anchorRect: rawRect,
        label: showConfidence
            ? '${detection.bboxLabel} ${(detection.confidence * 100).toStringAsFixed(1)}%'
            : detection.bboxLabel,
        color: Colors.amberAccent,
      );

      final rotatedRect = detection.rotatedRect;
      if (rotatedRect != null) {
        final path = _buildRotatedRectPath(rotatedRect, size);
        canvas.drawPath(path, roiPaint);
        _paintLabel(
          canvas,
          anchorRect: _rotatedRectBounds(rotatedRect, size),
          label: detection.roiLabel,
          color: Colors.lightGreenAccent,
        );
      }
    }
  }

  void _paintLabel(
    Canvas canvas, {
    required Rect anchorRect,
    required String label,
    required Color color,
  }) {
      final textSpan = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.black87,
          fontSize: 14,
          fontWeight: FontWeight.w600,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final textBackground = Rect.fromLTWH(
        anchorRect.left,
        math.max(0, anchorRect.top - textPainter.height - 4),
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final backgroundPaint = Paint()
        ..color = color.withValues(alpha: 0.85)
        ..style = PaintingStyle.fill;
      canvas.drawRect(textBackground, backgroundPaint);
      textPainter.paint(
        canvas,
        Offset(textBackground.left + 4, textBackground.top + 2),
      );
  }

  Rect _maybeMirror(Rect box) {
    if (Platform.isIOS || lensDirection != CameraLensDirection.front) {
      return box;
    }
    return Rect.fromLTRB(
      (1 - box.right).clamp(0.0, 1.0),
      box.top,
      (1 - box.left).clamp(0.0, 1.0),
      box.bottom,
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
    for (final corner in corners.skip(1)) {
      minX = math.min(minX, corner.dx);
      minY = math.min(minY, corner.dy);
      maxX = math.max(maxX, corner.dx);
      maxY = math.max(maxY, corner.dy);
    }
    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  List<Offset> _rotatedRectCorners(NormalizedRect rect, Size size) {
    final double centerX = rect.xCenter * size.width;
    final double centerY = rect.yCenter * size.height;
    final double width = rect.width * size.width;
    final double height = rect.height * size.height;
    final double cosR = math.cos(rect.rotation);
    final double sinR = math.sin(rect.rotation);
    final List<Offset> localCorners = <Offset>[
      Offset(-width * 0.5, -height * 0.5),
      Offset(width * 0.5, -height * 0.5),
      Offset(width * 0.5, height * 0.5),
      Offset(-width * 0.5, height * 0.5),
    ];
    final List<Offset> corners = localCorners.map((corner) {
      final double rotatedX = cosR * corner.dx - sinR * corner.dy + centerX;
      final double rotatedY = sinR * corner.dx + cosR * corner.dy + centerY;
      return Offset(rotatedX, rotatedY);
    }).toList();
    if (Platform.isIOS || lensDirection != CameraLensDirection.front) {
      return corners;
    }
    return corners
        .map((corner) => Offset(size.width - corner.dx, corner.dy))
        .toList();
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.lensDirection != lensDirection ||
        oldDelegate.showConfidence != showConfidence;
  }
}
