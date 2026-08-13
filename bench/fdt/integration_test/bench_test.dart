
// Android-only bench: import the native branch directly so the analyzer
// resolves the native create() signature (the public entry point's default
// branch is the web implementation, which lacks `accelerators`).
// ignore: implementation_imports
import 'package:face_detection_tflite/src/native/face_native_lib.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'bench_util.dart';

const String kApp = 'face_detection_tflite';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('single image', () {
    for (final MapEntry<String, Set<Accelerator>> accel
        in <String, Set<Accelerator>>{
          'default': <Accelerator>{Accelerator.gpu, Accelerator.cpu},
          'cpuOnly': <Accelerator>{Accelerator.cpu},
        }.entries) {
      for (final FaceDetectionMode mode in <FaceDetectionMode>[
        FaceDetectionMode.fast,
        FaceDetectionMode.standard,
        FaceDetectionMode.full,
      ]) {
        testWidgets('accel=${accel.key} mode=$mode', (
          WidgetTester tester,
        ) async {
          final FaceDetector detector = await FaceDetector.create(
            accelerators: accel.value,
          );

        for (final String asset in <String>['assets/portrait.jpg']) {
          final ByteData data = await rootBundle.load(asset);
          final Uint8List bytes = data.buffer.asUint8List();

          for (int i = 0; i < kWarmupRuns; i++) {
            await detector.detectFacesFromBytes(bytes, mode: mode);
          }
          final List<double> samples = <double>[];
          final Stopwatch sw = Stopwatch();
          for (int i = 0; i < kMeasuredRuns; i++) {
            sw
              ..reset()
              ..start();
            final List<Face> faces = await detector.detectFacesFromBytes(
              bytes,
              mode: mode,
            );
            sw.stop();
            expect(faces, isNotEmpty, reason: 'no face detected');
            samples.add(sw.elapsedMicroseconds / 1000.0);
          }
          emitResult(
            app: kApp,
            suite: 'single_image',
            config: <String, Object?>{
              'image': asset.split('/').last.replaceAll('.jpg', ''),
              'mode': '$mode'.split('.').last,
              'accel': accel.key,
            },
            samplesMs: samples,
          );
        }
        });
      }
    }
  });
}
