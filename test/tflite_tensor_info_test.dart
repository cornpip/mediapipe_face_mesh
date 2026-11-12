import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/tflite_interpreter.dart';

void main() {
  test('elementCount is the product of the shape', () {
    const TfliteTensorInfo info = TfliteTensorInfo(
      name: 'input',
      type: TfliteTensorType.float32,
      shape: <int>[1, 192, 256, 3],
      byteSize: 1 * 192 * 256 * 3 * 4,
    );

    expect(info.elementCount, 192 * 256 * 3);
    expect(info.elementCount * info.type.bytesPerElement, info.byteSize);
  });

  test('a scalar tensor has one element', () {
    const TfliteTensorInfo info = TfliteTensorInfo(
      name: 'score',
      type: TfliteTensorType.float32,
      shape: <int>[],
      byteSize: 4,
    );

    expect(info.elementCount, 1);
  });

  test('other has no element size', () {
    expect(TfliteTensorType.other.bytesPerElement, 0);
  });
}
