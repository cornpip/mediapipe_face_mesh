import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

void main() {
  test('distancePixels returns pixel distance between landmarks', () {
    final FaceMeshResult result = _resultWithLandmarks(<int, FaceMeshLandmark>{
      33: FaceMeshLandmark(x: 0.25, y: 0.5, z: 0),
      263: FaceMeshLandmark(x: 0.75, y: 0.5, z: 0),
    });

    expect(result.distancePixels(33, 263), closeTo(100, 1e-9));
  });

  test('estimateGeometry requires enough landmarks', () {
    final FaceMeshResult result = FaceMeshResult(
      landmarks: <FaceMeshLandmark>[FaceMeshLandmark(x: 0.25, y: 0.5, z: 0)],
      rect: const NormalizedRect(
        xCenter: 0.5,
        yCenter: 0.5,
        width: 1,
        height: 1,
      ),
      score: 1,
      imageWidth: 200,
      imageHeight: 100,
    );

    expect(result.estimateGeometry, throwsStateError);
  });
}

FaceMeshResult _resultWithLandmarks(Map<int, FaceMeshLandmark> overrides) {
  final List<FaceMeshLandmark> landmarks = List<FaceMeshLandmark>.generate(
    468,
    (int index) => FaceMeshLandmark(x: 0.5, y: 0.5, z: 0),
  );
  overrides.forEach((int index, FaceMeshLandmark landmark) {
    landmarks[index] = landmark;
  });
  return FaceMeshResult(
    landmarks: landmarks,
    rect: const NormalizedRect(xCenter: 0.5, yCenter: 0.5, width: 1, height: 1),
    score: 1,
    imageWidth: 200,
    imageHeight: 100,
  );
}
