import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

class FaceMeshPainter extends CustomPainter {
  FaceMeshPainter({
    required this.result,
    required this.rotationCompensation,
    required this.lensDirection,
    this.strokeColor = Colors.greenAccent,
    this.strokeWidth = 0.4,
    this.dotRadius = 1.5,
    this.drawDots = false,
  });

  final FaceMeshResult result;
  final int rotationCompensation;
  final CameraLensDirection lensDirection;
  final Color strokeColor;
  final double strokeWidth;
  final double dotRadius;
  final bool drawDots;

  @override
  void paint(Canvas canvas, Size size) {
    if (drawDots) {
      final paint = Paint()
        ..color = strokeColor
        ..style = PaintingStyle.fill;

      for (final lm in result.landmarks) {
        final p = _map(lm, size);
        canvas.drawCircle(p, dotRadius, paint);
      }
    } else {
      final paint = Paint()
        ..color = strokeColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth;

      for (final triangle in result.triangles) {
        final p0 = _map(triangle.points[0], size);
        final p1 = _map(triangle.points[1], size);
        final p2 = _map(triangle.points[2], size);

        final path = Path()
          ..moveTo(p0.dx, p0.dy)
          ..lineTo(p1.dx, p1.dy)
          ..lineTo(p2.dx, p2.dy)
          ..close();

        canvas.drawPath(path, paint);
      }
    }
  }

  Offset _map(FaceMeshLandmark lm, Size size) {
    return result.landmarkAsOffset(
      lm,
      targetSize: size,
      rotationDegrees: rotationCompensation,
      mirrorHorizontal:
          !Platform.isIOS && lensDirection == CameraLensDirection.front,
    );
  }

  @override
  bool shouldRepaint(covariant FaceMeshPainter oldDelegate) {
    return oldDelegate.result != result ||
        oldDelegate.rotationCompensation != rotationCompensation ||
        oldDelegate.lensDirection != lensDirection ||
        oldDelegate.strokeColor != strokeColor ||
        oldDelegate.strokeWidth != strokeWidth ||
        oldDelegate.dotRadius != dotRadius ||
        oldDelegate.drawDots != drawDots;
  }
}
