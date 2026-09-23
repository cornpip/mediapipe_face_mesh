import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

void main() {
  FaceMeshResult result(int count) => FaceMeshResult(
    landmarks: List<FaceMeshLandmark>.generate(
      count,
      (_) => FaceMeshLandmark(x: 0, y: 0, z: 0),
    ),
    rect: const NormalizedRect(xCenter: 0.5, yCenter: 0.5, width: 1, height: 1),
    score: 1,
    imageWidth: 1,
    imageHeight: 1,
  );

  test('hasIris is true only with the ten iris landmarks', () {
    expect(result(468).hasIris, isFalse);
    expect(result(478).hasIris, isTrue);
  });
}
