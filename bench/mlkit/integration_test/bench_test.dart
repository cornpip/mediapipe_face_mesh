import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_mesh_detection/google_mlkit_face_mesh_detection.dart';
import 'package:integration_test/integration_test.dart';

import 'bench_util.dart';

const String kApp = 'google_mlkit_face_mesh_detection';

/// nv21 input: decode + convert happen here, outside the measured loop, so
/// the measured call is detection + mesh only. Odd dimensions are cropped
/// by one pixel; NV21 needs even width/height.
Future<InputImage> nv21Asset(String assetPath) async {
  final ByteData data = await rootBundle.load(assetPath);
  final ui.Codec codec =
      await ui.instantiateImageCodec(data.buffer.asUint8List());
  final ui.FrameInfo frame = await codec.getNextFrame();
  final ui.Image image = frame.image;
  final ByteData? rgba =
      await image.toByteData(format: ui.ImageByteFormat.rawRgba);
  final int srcWidth = image.width;
  final int width = image.width & ~1;
  final int height = image.height & ~1;
  final Uint8List px = rgba!.buffer.asUint8List();

  final Uint8List nv21 = Uint8List(width * height + width * height ~/ 2);
  int vuIndex = width * height;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int p = (y * srcWidth + x) * 4;
      final int r = px[p], g = px[p + 1], b = px[p + 2];
      nv21[y * width + x] =
          (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(0, 255);
      if (y.isEven && x.isEven) {
        nv21[vuIndex++] =
            (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(0, 255);
        nv21[vuIndex++] =
            (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(0, 255);
      }
    }
  }
  image.dispose();

  return InputImage.fromBytes(
    bytes: nv21,
    metadata: InputImageMetadata(
      size: ui.Size(width.toDouble(), height.toDouble()),
      rotation: InputImageRotation.rotation0deg,
      format: InputImageFormat.nv21,
      bytesPerRow: width,
    ),
  );
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // FaceMeshDetectorOptions.faceMesh: one call does detection plus the 468
  // point mesh, the same unit of work as our detector + mesh pipeline.
  group('single image', () {
    testWidgets('mode=faceMesh input=nv21', (WidgetTester tester) async {
      await thermalCooldown();

      final FaceMeshDetector detector = FaceMeshDetector(
        option: FaceMeshDetectorOptions.faceMesh,
      );

      for (final String asset in <String>['assets/portrait.jpg']) {
        final InputImage image = await nv21Asset(asset);

        for (int i = 0; i < kWarmupRuns; i++) {
          await detector.processImage(image);
        }
        final List<double> samples = <double>[];
        final Stopwatch sw = Stopwatch();
        for (int i = 0; i < kMeasuredRuns; i++) {
          sw
            ..reset()
            ..start();
          final List<FaceMesh> meshes = await detector.processImage(image);
          sw.stop();
          expect(meshes, isNotEmpty, reason: 'no face mesh detected');
          samples.add(sw.elapsedMicroseconds / 1000.0);
        }
        emitResult(
          app: kApp,
          suite: 'single_image',
          config: <String, Object?>{
            'image': asset.split('/').last.replaceAll('.jpg', ''),
            'mode': 'faceMesh',
            'input': 'nv21',
          },
          samplesMs: samples,
        );
      }

      await detector.close();
    });
  });

  group('streaming', () {
    // ML Kit face mesh detection has no tracking mode, so every frame pays
    // the full detection + mesh pass. Frames are decoded and converted to
    // nv21 outside the stopwatch, one at a time.
    testWidgets('mode=faceMesh input=nv21', (WidgetTester tester) async {
      await thermalCooldown();

      final FaceMeshDetector detector = FaceMeshDetector(
        option: FaceMeshDetectorOptions.faceMesh,
      );

      final AssetManifest manifest =
          await AssetManifest.loadFromAssetBundle(rootBundle);
      final List<String> framePaths = manifest
          .listAssets()
          .where(
            (String p) => p.startsWith('assets/frames/') && p.endsWith('.jpg'),
          )
          .toList()
        ..sort();
      expect(framePaths, isNotEmpty,
          reason: 'run tool/prepare_assets.py first');

      final List<double> samples = <double>[];
      final Stopwatch sw = Stopwatch();
      int width = 0, height = 0;
      int framesNoFace = 0;
      for (int i = 0; i < framePaths.length; i++) {
        final InputImage image = await nv21Asset(framePaths[i]);
        width = image.metadata!.size.width.toInt();
        height = image.metadata!.size.height.toInt();
        sw
          ..reset()
          ..start();
        final List<FaceMesh> meshes = await detector.processImage(image);
        sw.stop();
        // A missed frame still costs a full pass; keep its latency sample
        // and report the miss count alongside.
        if (meshes.isEmpty) {
          framesNoFace++;
        }
        if (i >= kStreamSettleFrames) {
          samples.add(sw.elapsedMicroseconds / 1000.0);
        }
      }
      expect(framesNoFace, lessThan(framePaths.length ~/ 2),
          reason: 'detector missed most frames');
      emitResult(
        app: kApp,
        suite: 'streaming',
        config: <String, Object?>{
          'frames': framePaths.length,
          'width': width,
          'height': height,
          'mode': 'faceMesh',
          'input': 'nv21',
          'tracking': 'not supported',
        },
        samplesMs: samples,
        extra: <String, Object?>{'framesNoFace': framesNoFace},
      );

      await detector.close();
    });
  });
}
