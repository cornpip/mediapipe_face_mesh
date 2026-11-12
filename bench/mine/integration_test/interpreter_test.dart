import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';
import 'package:mediapipe_face_mesh/tflite_interpreter.dart';

/// Functional exercises for the generic interpreter, run on the bundled
/// blendshapes model (146x2 float32 landmarks in, 52 float32 scores out) and
/// on three one-operator models that add integers.

/// int32, shape [1, 3] with a dynamic second dimension. Tensors `x` and
/// `sum`. Built by tool/make_interpreter_test_models.py.
final Uint8List _int32AddModel = base64Decode(
  'CAAAAFRGTDPC////VAEAABAAAAAoAAAAPAEAAAMAAAAWAAAAaW50ZXJwcmV0ZXIgdGVzdCBtb2RlbAAAAQAAABQAAAAAAA4AGAAUABAADAAIAAQADgAAABQAAAAsAAAAGAAAABwAAABkAAAABAAAAG1haW4AAAAAAQAAAAEAAAABAAAAAAAAAAEAAAAUAAAAAAAOABQAAAAQAAwACwAEAA4AAAAkAAAAAAAACwgAAAAMAAAAAQAAAAEAAAACAAAAAAAAAAAAAABg////AgAAAFAAAAAEAAAAzP///xAAAAAkAAAAAAAAAhAAAAACAAAAAQAAAP////8CAAAAAQAAAAMAAAADAAAAc3VtABQAFAAQAA8AAAAIAAAAAAAAAAQAFAAAABAAAAAkAAAAAAAAAhAAAAACAAAAAQAAAP////8CAAAAAQAAAAMAAAABAAAAeAAAAAEAAAAEAAAA9P///wEAAAAIAAAABAAEAAQAAAA=',
);

/// uint8, same graph. Input quantization (0.5, 3), output (1.0, 6), so the
/// quantized output is the quantized input plus 3.
final Uint8List _uint8AddModel = base64Decode(
  'CAAAAFRGTDPC////rAEAABAAAAAoAAAAlAEAAAMAAAAWAAAAaW50ZXJwcmV0ZXIgdGVzdCBtb2RlbAAAAQAAABQAAAAAAA4AGAAUABAADAAIAAQADgAAABQAAAAsAAAAGAAAABwAAABkAAAABAAAAG1haW4AAAAAAQAAAAEAAAABAAAAAAAAAAEAAAAUAAAAAAAOABQAAAAQAAwACwAEAA4AAAAkAAAAAAAACwgAAAAMAAAAAQAAAAEAAAACAAAAAAAAAAAAAAAI////AgAAAHQAAAAEAAAAqP///zQAAAAQAAAARAAAAAAAAAMwAAAAlP///wgAAAAQAAAAAQAAAAYAAAAAAAAAAQAAAAAAgD8CAAAAAQAAAP////8CAAAAAQAAAAMAAAADAAAAc3VtABQAGAAUABMAAAAMAAgAAAAAAAQAFAAAAEQAAAAcAAAAVAAAAAAAAANAAAAADAAMAAAAAAAIAAQADAAAAAgAAAAUAAAAAQAAAAMAAAAAAAAAAAAAAAEAAAAAAAA/AgAAAAEAAAD/////AgAAAAEAAAADAAAAAQAAAHgAAAABAAAABAAAAPT///8BAAAACAAAAAQABAAEAAAA',
);

/// int32, two inputs `x` and `y` of shape [1, 3] with a dynamic second
/// dimension, added into `sum`. The inputs must be resized together.
final Uint8List _int32AddTwoModel = base64Decode(
  'CAAAAFRGTDPC////kAEAABAAAAAoAAAAeAEAAAMAAAAWAAAAaW50ZXJwcmV0ZXIgdGVzdCBtb2RlbAAAAQAAABQAAAAAAA4AGAAUABAADAAIAAQADgAAABQAAAAwAAAAGAAAABwAAABoAAAABAAAAG1haW4AAAAAAQAAAAIAAAACAAAAAAAAAAEAAAABAAAAFAAAAAAADgAUAAAAEAAMAAsABAAOAAAAJAAAAAAAAAsIAAAADAAAAAEAAAACAAAAAgAAAAAAAAABAAAAKP///wMAAACIAAAAPAAAAAQAAACY////EAAAACQAAAAAAAACEAAAAAIAAAABAAAA/////wIAAAABAAAAAwAAAAMAAABzdW0AzP///xAAAAAkAAAAAAAAAhAAAAACAAAAAQAAAP////8CAAAAAQAAAAMAAAABAAAAeQAAABQAFAAQAA8AAAAIAAAAAAAAAAQAFAAAABAAAAAkAAAAAAAAAhAAAAACAAAAAQAAAP////8CAAAAAQAAAAMAAAABAAAAeAAAAAEAAAAEAAAA9P///wEAAAAIAAAABAAEAAQAAAA=',
);

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

  test('run takes and fills nested lists', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      modelBytes,
    );
    try {
      interpreter
        ..setInput(0, input)
        ..invoke();
      final Float32List expected = interpreter.output(0);

      // Nest the flat input the way the model shapes it.
      Object nest(List<int> shape, int offset) {
        if (shape.length == 1) {
          return input.sublist(offset, offset + shape.first).toList();
        }
        final int stride = shape.skip(1).reduce((int a, int b) => a * b);
        return <Object>[
          for (int i = 0; i < shape.first; i++)
            nest(shape.sublist(1), offset + i * stride),
        ];
      }

      final List<int> inputShape = interpreter.inputs.single.shape;
      final List<int> outputShape = interpreter.outputs.first.shape;
      Object zeros(List<int> shape) => shape.length == 1
          ? List<double>.filled(shape.first, 0)
          : <Object>[
              for (int i = 0; i < shape.first; i++) zeros(shape.sublist(1)),
            ];
      final Object output = zeros(outputShape);

      interpreter.run(nest(inputShape, 0), output);

      Object leaf = output;
      while (leaf is List && leaf.first is List) {
        leaf = leaf.first as Object;
      }
      expect(leaf, expected);
      // The shape did not change, so run must not have resized the input.
      expect(interpreter.inputs.single.shape, inputShape);
    } finally {
      interpreter.close();
    }
  });

  test('integer tensors use typed lists of their own type', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      _int32AddModel,
    );
    try {
      final TfliteTensorInfo info = interpreter.inputs.single;
      expect(info.type, TfliteTensorType.int32);
      expect(info.shape, <int>[1, 3]);
      expect(info.quantization, isNull);
      expect(interpreter.inputIndex('x'), 0);
      expect(interpreter.outputIndex('sum'), 0);
      expect(() => interpreter.inputIndex('nope'), throwsArgumentError);

      final List<List<int>> output = <List<int>>[
        <int>[0, 0, 0],
      ];
      interpreter.run(<List<int>>[
        <int>[1, -2, 3],
      ], output);
      expect(output, <List<int>>[
        <int>[2, -4, 6],
      ]);
      expect(interpreter.outputData(0), isA<Int32List>());
      expect(interpreter.outputData(0), <int>[2, -4, 6]);

      interpreter
        ..setInput(0, Int32List.fromList(<int>[10, 20, 30]))
        ..invoke();
      expect(interpreter.outputData(0), <int>[20, 40, 60]);

      // A typed list of another type is rejected instead of reinterpreted.
      expect(
        () => interpreter.setInput(0, Float32List(3)),
        throwsArgumentError,
      );
      expect(() => interpreter.output(0), throwsArgumentError);
    } finally {
      interpreter.close();
    }
  });

  test('inputs resize explicitly and through run', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      _int32AddModel,
    );
    try {
      final List<List<int>> output = <List<int>>[List<int>.filled(5, 0)];
      interpreter.run(<List<int>>[
        <int>[1, 2, 3, 4, 5],
      ], output);
      expect(output.single, <int>[2, 4, 6, 8, 10]);
      expect(interpreter.inputs.single.shape, <int>[1, 5]);
      expect(interpreter.outputs.single.shape, <int>[1, 5]);

      interpreter.resizeInput(0, <int>[1, 2]);
      expect(interpreter.inputs.single.shape, <int>[1, 2]);
      interpreter
        ..setInput(0, <int>[7, 8])
        ..invoke();
      expect(interpreter.outputData(0), <int>[14, 16]);

      expect(
        () => interpreter.setInput(0, <int>[1, 2, 3]),
        throwsArgumentError,
      );
      expect(
        () => interpreter.copyOutputTo(0, <List<int>>[
          <int>[0, 0, 0],
        ]),
        throwsArgumentError,
      );
    } finally {
      interpreter.close();
    }
  });

  test('quantized tensors report their parameters', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      _uint8AddModel,
    );
    try {
      final TfliteQuantization input = interpreter.inputs.single.quantization!;
      final TfliteQuantization output =
          interpreter.outputs.single.quantization!;
      expect(interpreter.inputs.single.type, TfliteTensorType.uint8);
      expect(input.scale, 0.5);
      expect(input.zeroPoint, 3);
      expect(output.scale, 1.0);
      expect(output.zeroPoint, 6);

      // Real 0, 1, 5 in; real 0, 2, 10 out.
      interpreter
        ..setInput(
          0,
          Uint8List.fromList(<int>[
            for (final double real in <double>[0, 1, 5]) input.quantize(real),
          ]),
        )
        ..invoke();
      final Uint8List quantized = interpreter.outputData(0) as Uint8List;
      expect(quantized.map(output.dequantize), <double>[0, 2, 10]);
    } finally {
      interpreter.close();
    }
  });

  test('run resizes every input before it reallocates', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      _int32AddTwoModel,
    );
    try {
      expect(interpreter.inputs, hasLength(2));
      final List<List<int>> output = <List<int>>[List<int>.filled(4, 0)];
      // Both inputs grow from 3 to 4 columns. Allocating after only the
      // first one changed would fail, because 4 and 3 columns do not add.
      interpreter.runForMultipleInputs(
        <Object>[
          <List<int>>[
            <int>[1, 2, 3, 4],
          ],
          <List<int>>[
            <int>[10, 20, 30, 40],
          ],
        ],
        <int, Object>{0: output},
      );
      expect(output.single, <int>[11, 22, 33, 44]);
    } finally {
      interpreter.close();
    }
  });

  test('a failed allocation goes back to the previous shapes', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      _int32AddTwoModel,
    );
    try {
      // Only one of the two inputs is resized, so the shapes cannot add.
      interpreter.resizeInput(0, <int>[1, 4]);
      expect(interpreter.invoke, throwsA(isA<FaceMeshException>()));

      expect(interpreter.inputs.first.shape, <int>[1, 3]);
      interpreter
        ..setInput(0, <int>[1, 2, 3])
        ..setInput(1, <int>[4, 5, 6])
        ..invoke();
      expect(interpreter.outputData(0), <int>[5, 7, 9]);
    } finally {
      interpreter.close();
    }
  });

  test('values that do not fit are rejected without side effects', () {
    final TfliteInterpreter interpreter = TfliteInterpreter.fromBuffer(
      _int32AddModel,
    );
    try {
      interpreter.setInput(0, <int>[1, 2, 3]);

      // Wrong length, a fraction, and NaN for an integer tensor. None of
      // them may overwrite the input that is already there.
      expect(
        () => interpreter.setInput(0, <int>[9, 9, 9, 9]),
        throwsArgumentError,
      );
      expect(
        () => interpreter.setInput(0, <num>[9, 9.5, 9]),
        throwsArgumentError,
      );
      expect(
        () => interpreter.setInput(0, <num>[9, double.nan, 9]),
        throwsArgumentError,
      );
      interpreter.invoke();
      expect(interpreter.outputData(0), <int>[2, 4, 6]);

      // A whole-number double is fine.
      interpreter
        ..setInput(0, <num>[1.0, 2, 3])
        ..invoke();
      expect(interpreter.outputData(0), <int>[2, 4, 6]);

      // A flat list of another length cannot stand in for shape [1, 3], so
      // it is rejected instead of resizing the input to one dimension.
      expect(
        () => interpreter.run(
          <int>[1, 2, 3, 4, 5],
          <List<int>>[
            <int>[0, 0, 0, 0, 0],
          ],
        ),
        throwsArgumentError,
      );
      expect(interpreter.inputs.single.shape, <int>[1, 3]);

      // Output targets of another element type.
      expect(
        () => interpreter.copyOutputTo(0, <List<double>>[
          <double>[0, 0, 0],
        ]),
        throwsArgumentError,
      );
      expect(
        () => interpreter.copyOutputTo(0, Float32List(3)),
        throwsArgumentError,
      );
      final Int32List typed = Int32List(3);
      interpreter.copyOutputTo(0, typed);
      expect(typed, <int>[2, 4, 6]);
    } finally {
      interpreter.close();
    }
  });
}
