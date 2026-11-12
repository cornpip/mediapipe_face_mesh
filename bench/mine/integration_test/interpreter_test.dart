import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';
import 'package:mediapipe_face_mesh/tflite_interpreter.dart';

/// Functional exercises for the generic interpreter, run on the bundled
/// blendshapes model (146x2 float32 landmarks in, 52 float32 scores out).
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List modelBytes;
  late Float32List input;

  setUpAll(() async {
    final ByteData data = await rootBundle.load(
      'packages/mediapipe_face_mesh/assets/models/face_blendshapes.tflite',
    );
    modelBytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    // Landmark-like pixel coordinates; the values only need to be stable.
    input = Float32List.fromList(<double>[
      for (int i = 0; i < 146 * 2; i++) 64.0 + (i * 37 % 128),
    ]);
  });

  test('reports tensor info and runs a model loaded from bytes', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      modelBytes,
    );
    try {
      expect(interpreter.inputs, hasLength(1));
      final TfliteTensorInfo inputInfo = interpreter.inputs.single;
      expect(inputInfo.type, TfliteTensorType.float32);
      expect(inputInfo.elementCount, 146 * 2);
      expect(inputInfo.byteSize, 146 * 2 * 4);
      expect(interpreter.activeDelegate, FaceMeshDelegate.cpu);

      interpreter.setInput(0, input);
      interpreter.invoke();

      final Float32List scores = interpreter.output(0);
      expect(scores, hasLength(52));
      expect(scores.every((double v) => v.isFinite), isTrue);
      expect(interpreter.outputs.first.elementCount, 52);
      expect(interpreter.outputBytes(0), hasLength(52 * 4));

      // The model bytes were copied: a second run still works and repeats.
      interpreter.invoke();
      expect(interpreter.output(0), scores);
    } finally {
      interpreter.close();
    }
  });

  test('a model loaded from a file matches the same model from bytes', () {
    final Directory dir = Directory.systemTemp.createTempSync('mpfm_interp');
    final File file = File('${dir.path}/model.tflite')
      ..writeAsBytesSync(modelBytes);
    final TfliteInterpreter fromBuffer = TfliteInterpreter.fromBuffer(
      modelBytes,
    );
    final TfliteInterpreter fromFile = TfliteInterpreter.fromFile(file.path);
    try {
      fromBuffer
        ..setInput(0, input)
        ..invoke();
      fromFile
        ..setInputBytes(0, input.buffer.asUint8List())
        ..invoke();
      expect(fromFile.output(0), fromBuffer.output(0));
    } finally {
      fromBuffer.close();
      fromFile.close();
      dir.deleteSync(recursive: true);
    }
  });

  test('the XNNPACK delegate agrees with the CPU path', () {
    final TfliteInterpreter cpu = TfliteInterpreter.fromBuffer(modelBytes);
    final TfliteInterpreter xnnpack = TfliteInterpreter.fromBuffer(
      modelBytes,
      delegate: FaceMeshDelegate.xnnpack,
      threads: 2,
    );
    try {
      cpu
        ..setInput(0, input)
        ..invoke();
      xnnpack
        ..setInput(0, input)
        ..invoke();
      final Float32List expected = cpu.output(0);
      final Float32List actual = xnnpack.output(0);
      for (int i = 0; i < expected.length; i++) {
        expect(actual[i], closeTo(expected[i], 1e-3), reason: 'score $i');
      }
    } finally {
      cpu.close();
      xnnpack.close();
    }
  });

  test('rejects bad models, bad inputs, and use after close', () {
    expect(
      () => TfliteInterpreter.fromBuffer(Uint8List.fromList(<int>[1, 2, 3, 4])),
      throwsA(isA<FaceMeshException>()),
    );
    expect(
      () => TfliteInterpreter.fromFile('does/not/exist.tflite'),
      throwsA(isA<FaceMeshException>()),
    );
    expect(
      () => TfliteInterpreter.fromBuffer(Uint8List(0)),
      throwsArgumentError,
    );
    expect(
      () => TfliteInterpreter.fromBuffer(modelBytes, threads: 0),
      throwsArgumentError,
    );

    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      modelBytes,
    );
    expect(() => interpreter.setInput(0, Float32List(3)), throwsArgumentError);
    expect(() => interpreter.setInput(1, input), throwsRangeError);
    expect(() => interpreter.output(9), throwsRangeError);
    interpreter.close();
    interpreter.close();
    expect(interpreter.invoke, throwsStateError);
    expect(() => interpreter.outputs, throwsStateError);
  });
}
