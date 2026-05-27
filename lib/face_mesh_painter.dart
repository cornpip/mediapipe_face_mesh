import 'package:flutter/material.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Draws one or more [FaceMeshResult] overlays on a [CustomPaint] canvas.
///
/// Provide the same [rotationDegrees] and [mirrorHorizontal] values used for
/// preview mapping. This class intentionally does not depend on the camera
/// package so it can be used with any image source.
class FaceMeshPainter extends CustomPainter {
  /// Creates a painter for one or more face mesh results.
  FaceMeshPainter({
    FaceMeshResult? result,
    List<FaceMeshResult>? results,
    this.rotationDegrees = 0,
    this.mirrorHorizontal = false,
    this.strokeColor = Colors.greenAccent,
    this.irisColor = Colors.cyanAccent,
    this.refinedEyeColor = Colors.greenAccent,
    this.strokeWidth = 0.4,
    this.dotRadius = 1.5,
    this.irisDotRadius = 1.8,
    this.drawDots = false,
    this.drawRefinedEyeEdges = true,
    this.drawIris = true,
    this.clampToBounds = true,
  }) : results = _resolveResults(result, results);

  /// Mesh results to draw.
  final List<FaceMeshResult> results;

  /// Clockwise rotation applied when mapping normalized landmarks to pixels.
  final int rotationDegrees;

  /// Whether landmark X coordinates should be mirrored for preview overlays.
  final bool mirrorHorizontal;

  /// Color used for the base face mesh.
  final Color strokeColor;

  /// Color used for appended iris landmarks.
  final Color irisColor;

  /// Color used for refined eye-region edges.
  final Color refinedEyeColor;

  /// Width of mesh strokes.
  final double strokeWidth;

  /// Radius used when [drawDots] is true.
  final double dotRadius;

  /// Radius used for appended iris landmarks.
  final double irisDotRadius;

  /// Draws landmarks as dots instead of triangle wireframes.
  final bool drawDots;

  /// Draws refined eye-region edges when iris refinement is available.
  final bool drawRefinedEyeEdges;

  /// Draws appended iris landmarks when iris refinement is available.
  final bool drawIris;

  /// Clamps mapped landmark coordinates to the paint bounds.
  final bool clampToBounds;

  static List<FaceMeshResult> _resolveResults(
    FaceMeshResult? result,
    List<FaceMeshResult>? results,
  ) {
    if ((result == null) == (results == null)) {
      throw ArgumentError('Provide exactly one of result or results.');
    }
    return results ?? <FaceMeshResult>[result!];
  }

  @override
  void paint(Canvas canvas, Size size) {
    for (final FaceMeshResult result in results) {
      _paintResult(canvas, size, result);
    }
  }

  void _paintResult(Canvas canvas, Size size, FaceMeshResult result) {
    if (drawDots) {
      final Paint paint = Paint()
        ..color = strokeColor
        ..style = PaintingStyle.fill;

      for (final FaceMeshLandmark landmark in result.landmarks) {
        canvas.drawCircle(_map(result, landmark, size), dotRadius, paint);
      }
    } else {
      final Paint paint = Paint()
        ..color = strokeColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth;

      for (final MpFaceMeshTriangle triangle in result.triangles) {
        final Offset p0 = _map(result, triangle.points[0], size);
        final Offset p1 = _map(result, triangle.points[1], size);
        final Offset p2 = _map(result, triangle.points[2], size);

        final Path path = Path()
          ..moveTo(p0.dx, p0.dy)
          ..lineTo(p1.dx, p1.dy)
          ..lineTo(p2.dx, p2.dy)
          ..close();

        canvas.drawPath(path, paint);
      }
    }

    if (drawRefinedEyeEdges) {
      _paintRefinedEyeEdges(canvas, size, result);
    }
    if (drawIris) {
      _paintIrisDots(canvas, size, result);
    }
  }

  void _paintRefinedEyeEdges(Canvas canvas, Size size, FaceMeshResult result) {
    if (result.landmarks.length <= 468) {
      return;
    }
    final Paint paint = Paint()
      ..color = refinedEyeColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;
    for (final MpFaceMeshTriangle triangle in result.triangles) {
      _drawRefinedEyeEdge(canvas, size, result, paint, triangle, 0, 1);
      _drawRefinedEyeEdge(canvas, size, result, paint, triangle, 1, 2);
      _drawRefinedEyeEdge(canvas, size, result, paint, triangle, 2, 0);
    }
  }

  void _drawRefinedEyeEdge(
    Canvas canvas,
    Size size,
    FaceMeshResult result,
    Paint paint,
    MpFaceMeshTriangle triangle,
    int from,
    int to,
  ) {
    final int fromIndex = triangle.indices[from];
    final int toIndex = triangle.indices[to];
    if (!faceMeshIrisRefinedEyeLandmarkIndices.contains(fromIndex) ||
        !faceMeshIrisRefinedEyeLandmarkIndices.contains(toIndex)) {
      return;
    }
    canvas.drawLine(
      _map(result, triangle.points[from], size),
      _map(result, triangle.points[to], size),
      paint,
    );
  }

  void _paintIrisDots(Canvas canvas, Size size, FaceMeshResult result) {
    if (result.landmarks.length <= 468) {
      return;
    }
    final Paint paint = Paint()
      ..color = irisColor
      ..style = PaintingStyle.fill;
    final int end = result.landmarks.length < 478
        ? result.landmarks.length
        : 478;
    for (int i = 468; i < end; i++) {
      canvas.drawCircle(
        _map(result, result.landmarks[i], size),
        irisDotRadius,
        paint,
      );
    }
  }

  Offset _map(FaceMeshResult result, FaceMeshLandmark landmark, Size size) {
    return result.landmarkAsOffset(
      landmark,
      targetSize: size,
      clampToBounds: clampToBounds,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
    );
  }

  @override
  bool shouldRepaint(covariant FaceMeshPainter oldDelegate) {
    return oldDelegate.results != results ||
        oldDelegate.rotationDegrees != rotationDegrees ||
        oldDelegate.mirrorHorizontal != mirrorHorizontal ||
        oldDelegate.strokeColor != strokeColor ||
        oldDelegate.irisColor != irisColor ||
        oldDelegate.refinedEyeColor != refinedEyeColor ||
        oldDelegate.strokeWidth != strokeWidth ||
        oldDelegate.dotRadius != dotRadius ||
        oldDelegate.irisDotRadius != irisDotRadius ||
        oldDelegate.drawDots != drawDots ||
        oldDelegate.drawRefinedEyeEdges != drawRefinedEyeEdges ||
        oldDelegate.drawIris != drawIris ||
        oldDelegate.clampToBounds != clampToBounds;
  }
}
