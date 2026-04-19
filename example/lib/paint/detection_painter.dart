import 'dart:io' show Platform;
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class Detection {
  const Detection({
    required this.boundingBox,
    required this.confidence,
    required this.label,
  });

  final Rect boundingBox;
  final double confidence;
  final String label;
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
    final boxPaint = Paint()
      ..color = Colors.lightGreenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (final detection in detections) {
      final mirroredBox = _maybeMirror(detection.boundingBox);
      final rect = Rect.fromLTRB(
        mirroredBox.left * size.width,
        mirroredBox.top * size.height,
        mirroredBox.right * size.width,
        mirroredBox.bottom * size.height,
      );
      canvas.drawRect(rect, boxPaint);

      final label = showConfidence
          ? '${detection.label} ${(detection.confidence * 100).toStringAsFixed(1)}%'
          : detection.label;
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
        rect.left,
        math.max(0, rect.top - textPainter.height - 4),
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final backgroundPaint = Paint()
        ..color = Colors.lightGreenAccent.withValues(alpha: 0.85)
        ..style = PaintingStyle.fill;
      canvas.drawRect(textBackground, backgroundPaint);
      textPainter.paint(
        canvas,
        Offset(textBackground.left + 4, textBackground.top + 2),
      );
    }
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

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.lensDirection != lensDirection ||
        oldDelegate.showConfidence != showConfidence;
  }
}
