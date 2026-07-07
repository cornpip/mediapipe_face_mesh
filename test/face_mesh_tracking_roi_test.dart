import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Golden tests for [FaceMeshResult.trackingRoi].
///
/// The expected values are hand-computed from the tracking ROI formula
/// (landmark bbox long side x 1.5 pixel-space square, eye-line rotation,
/// aspect-preserving size clamp). The same formula lives natively in
/// `src/mediapipe_face_mesh.cc` (`RectFromLandmarks` + `SanitizeRect`); if a
/// test here starts failing after a formula change, update both
/// implementations together.
void main() {
  FaceMeshResult resultWithBbox({
    required double minX,
    required double maxX,
    required double minY,
    required double maxY,
    required int imageWidth,
    required int imageHeight,
    double? eye33X,
    double? eye33Y,
    double? eye263X,
    double? eye263Y,
  }) {
    final double centerX = (minX + maxX) / 2;
    final double centerY = (minY + maxY) / 2;
    final List<FaceMeshLandmark> landmarks = List<FaceMeshLandmark>.generate(
      478,
      (int i) => FaceMeshLandmark(x: centerX, y: centerY, z: 0),
    );
    landmarks[0] = FaceMeshLandmark(x: minX, y: minY, z: 0);
    landmarks[1] = FaceMeshLandmark(x: maxX, y: maxY, z: 0);
    landmarks[33] = FaceMeshLandmark(
      x: eye33X ?? centerX - 0.01,
      y: eye33Y ?? centerY,
      z: 0,
    );
    landmarks[263] = FaceMeshLandmark(
      x: eye263X ?? centerX + 0.01,
      y: eye263Y ?? centerY,
      z: 0,
    );
    return FaceMeshResult(
      landmarks: landmarks,
      rect: const NormalizedRect(
        xCenter: 0.5,
        yCenter: 0.5,
        width: 1,
        height: 1,
      ),
      score: 1,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      triangles: const <MpFaceMeshTriangle>[],
    );
  }

  test('builds a pixel-space square on a portrait frame', () {
    // bbox 144x256 px on 720x1280 -> long side 256 * 1.5 = 384 px.
    final NormalizedRect roi = resultWithBbox(
      minX: 0.4,
      maxX: 0.6,
      minY: 0.3,
      maxY: 0.5,
      imageWidth: 720,
      imageHeight: 1280,
    ).trackingRoi();

    expect(roi.xCenter, closeTo(0.5, 1e-6));
    expect(roi.yCenter, closeTo(0.4, 1e-6));
    expect(roi.width, closeTo(384 / 720, 1e-6));
    expect(roi.height, closeTo(384 / 1280, 1e-6));
    expect(roi.rotation, closeTo(0, 1e-6));
    // Pixel-space square: width and height map to the same pixel length.
    expect(roi.width * 720, closeTo(roi.height * 1280, 1e-3));
  });

  test('rotation follows the eye line with aspect correction', () {
    // Eyes: 33 at (0.45, 0.38), 263 at (0.55, 0.42) on 720x1280
    // -> the deltas must be aspect-corrected to pixels: dx = 72, dy = 51.2.
    final NormalizedRect roi = resultWithBbox(
      minX: 0.4,
      maxX: 0.6,
      minY: 0.3,
      maxY: 0.5,
      imageWidth: 720,
      imageHeight: 1280,
      eye33X: 0.45,
      eye33Y: 0.38,
      eye263X: 0.55,
      eye263Y: 0.42,
    ).trackingRoi();

    expect(roi.rotation, closeTo(math.atan2(51.2, 72), 1e-9));
  });

  test('size clamp preserves the aspect ratio for small faces', () {
    // bbox 14.4x25.6 px on 720x1280 -> long side 38.4 px
    // -> width 0.05333, height 0.03 -> short side clamps to 0.1 with one
    // scale factor: width 0.17778, height 0.1.
    final NormalizedRect roi = resultWithBbox(
      minX: 0.49,
      maxX: 0.51,
      minY: 0.49,
      maxY: 0.51,
      imageWidth: 720,
      imageHeight: 1280,
    ).trackingRoi();

    expect(roi.height, closeTo(0.1, 1e-6));
    expect(roi.width, closeTo(0.1 * 1280 / 720, 1e-6));
    // Still a pixel-space square after clamping.
    expect(roi.width * 720, closeTo(roi.height * 1280, 1e-3));
  });

  test('returns a full-frame rect without usable landmarks', () {
    final FaceMeshResult empty = FaceMeshResult(
      landmarks: const <FaceMeshLandmark>[],
      rect: const NormalizedRect(
        xCenter: 0.5,
        yCenter: 0.5,
        width: 1,
        height: 1,
      ),
      score: 0,
      imageWidth: 720,
      imageHeight: 1280,
      triangles: const <MpFaceMeshTriangle>[],
    );

    final NormalizedRect roi = empty.trackingRoi();
    expect(roi.xCenter, 0.5);
    expect(roi.yCenter, 0.5);
    expect(roi.width, 1);
    expect(roi.height, 1);
  });
}
