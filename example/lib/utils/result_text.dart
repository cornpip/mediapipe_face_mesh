import 'dart:math' as math;

import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Head pose and distances of a mesh result, for the overlay chip.
/// estimateGeometry is a native solve, so call this once per result.
String geometryTextOf(FaceMeshResult result) {
  try {
    final geometry = result.estimateGeometry();
    final pose = geometry.headPose;
    final measurements = geometry.measurements;
    final double innerEyePixels = result.distancePixels(133, 362);
    final StringBuffer buf = StringBuffer(
      'Yaw ${pose.yawDegrees.toStringAsFixed(0)}°  '
      'Pitch ${pose.pitchDegrees.toStringAsFixed(0)}°  '
      'Roll ${pose.rollDegrees.toStringAsFixed(0)}°\n',
    );
    final ipd = measurements.interpupillaryDistance;
    if (ipd != null) {
      buf.write('IPD ${ipd.valueCm.toStringAsFixed(1)}cm  ');
    }
    buf.write(
      'Inner eye ${measurements.eyeInnerDistance.valueCm.toStringAsFixed(1)}cm\n'
      'Inner eye ${innerEyePixels.toStringAsFixed(0)}px',
    );
    return buf.toString();
  } on Object {
    return 'Geometry unavailable';
  }
}

/// Maps the 52 blendshape coefficients to a coarse facial movement label.
///
/// Thresholds are illustrative starting points; tune per camera and lighting.
String detectMovement(FaceBlendshapes blendshapes) {
  double v(FaceBlendshape shape) => blendshapes[shape];
  final double smile =
      (v(FaceBlendshape.mouthSmileLeft) + v(FaceBlendshape.mouthSmileRight)) /
      2;
  final double blink = math.max(
    v(FaceBlendshape.eyeBlinkLeft),
    v(FaceBlendshape.eyeBlinkRight),
  );

  if (blink > 0.45) {
    return 'Blink';
  }
  if (v(FaceBlendshape.jawOpen) > 0.35) {
    return 'Mouth open';
  }
  if (smile > 0.4) {
    return 'Smile';
  }
  return 'Neutral';
}
