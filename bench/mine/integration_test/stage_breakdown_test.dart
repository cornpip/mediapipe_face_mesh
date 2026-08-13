import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'bench_util.dart';

const String kApp = 'mediapipe_face_mesh';

Future<FaceMeshImage> loadRgbaAsset(String assetPath) async {
  final ByteData data = await rootBundle.load(assetPath);
  final ui.Codec codec = await ui.instantiateImageCodec(
    data.buffer.asUint8List(),
  );
  final ui.FrameInfo frame = await codec.getNextFrame();
  final ByteData? rgba = await frame.image.toByteData(
    format: ui.ImageByteFormat.rawRgba,
  );
  final FaceMeshImage image = FaceMeshImage(
    pixels: rgba!.buffer.asUint8List(),
    width: frame.image.width,
    height: frame.image.height,
  );
  frame.image.dispose();
  codec.dispose();
  return image;
}

List<double> measure(void Function() body) {
  for (int i = 0; i < kWarmupRuns; i++) {
    body();
  }
  final List<double> samples = <double>[];
  final Stopwatch sw = Stopwatch();
  for (int i = 0; i < kMeasuredRuns; i++) {
    sw
      ..reset()
      ..start();
    body();
    sw.stop();
    samples.add(sw.elapsedMicroseconds / 1000.0);
  }
  return samples;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late FaceMeshImage portrait;

  setUpAll(() async {
    portrait = await loadRgbaAsset('assets/portrait.jpg');
  });

  for (final int threads in <int>[2, 4]) {
    testWidgets('stage breakdown xnnpack threads=$threads', (
      WidgetTester tester,
    ) async {
      final FaceDetectorProcessor detector = await FaceDetectorProcessor.create(
        delegate: FaceMeshDelegate.xnnpack,
        threads: threads,
      );
      final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
        enableRoiTracking: false,
        enableSmoothing: false,
        delegate: FaceMeshDelegate.xnnpack,
        threads: threads,
      );

      final FaceDetectionResult det = detector.process(portrait);
      final NormalizedRect roi = det.primaryDetection!.expandedFaceRect!;

      final Map<String, List<double>> suites = <String, List<double>>{
        'detector_only': measure(() => detector.process(portrait)),
        'mesh_only': measure(() => mesh.process(portrait, roi: roi)),
        'pipeline': measure(() {
          final FaceDetectionResult d = detector.process(portrait);
          mesh.process(portrait, roi: d.primaryDetection!.expandedFaceRect);
        }),
      };

      for (final MapEntry<String, List<double>> e in suites.entries) {
        emitResult(
          app: kApp,
          suite: 'stage_${e.key}',
          config: <String, Object?>{
            'image': 'portrait',
            'delegate': 'xnnpack',
            'threads': threads,
            'activeDelegate': mesh.activeDelegate.name,
          },
          samplesMs: e.value,
        );
      }

      mesh.close();
      detector.close();
    });
  }
}
