import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

void main() {
  test(
    'enableAttentionMesh throws UnsupportedError on Windows',
    () {
      expect(
        () => FaceMeshProcessor.create(enableAttentionMesh: true),
        throwsUnsupportedError,
      );
      expect(
        () => FaceMeshProcessor.createForMultiFace(enableAttentionMesh: true),
        throwsUnsupportedError,
      );
    },
    skip: !Platform.isWindows
        ? 'Attention mesh is only blocked on Windows.'
        : false,
  );
}
