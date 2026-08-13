import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'bench_util.dart';

const String kApp = 'mediapipe_face_mesh';

/// Delegate/model matrix. cpu is the package default; xnnpack runs a single
/// config only, to show parity with cpu. gpuV2 is deprecated and excluded.
const List<FaceMeshDelegate> kDelegates = <FaceMeshDelegate>[
  FaceMeshDelegate.cpu,
  FaceMeshDelegate.xnnpack,
];
const List<bool> kAttention = <bool>[false, true];

bool skipConfig(FaceMeshDelegate delegate, bool attention) =>
    delegate == FaceMeshDelegate.xnnpack && attention;

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

Future<List<FaceMeshImage>> loadFrameSequence() async {
  final AssetManifest manifest = await AssetManifest.loadFromAssetBundle(
    rootBundle,
  );
  final List<String> framePaths =
      manifest
          .listAssets()
          .where((String p) => p.startsWith('assets/frames/'))
          .toList()
        ..sort();
  final List<FaceMeshImage> frames = <FaceMeshImage>[];
  for (final String path in framePaths) {
    frames.add(await loadRgbaAsset(path));
  }
  return frames;
}

/// Mean per-landmark displacement between consecutive frames, in pixels.
double meanJitterPx(List<List<FaceMeshLandmark>> perFrame, int w, int h) {
  double total = 0;
  int count = 0;
  for (int i = 1; i < perFrame.length; i++) {
    final List<FaceMeshLandmark> a = perFrame[i - 1];
    final List<FaceMeshLandmark> b = perFrame[i];
    final int n = math.min(a.length, b.length);
    for (int j = 0; j < n; j++) {
      final double dx = (a[j].x - b[j].x) * w;
      final double dy = (a[j].y - b[j].y) * h;
      total += math.sqrt(dx * dx + dy * dy);
      count++;
    }
  }
  return count == 0 ? 0 : total / count;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late FaceMeshImage portrait;

  setUpAll(() async {
    portrait = await loadRgbaAsset('assets/portrait.jpg');
  });

  group('single image (cold pipeline: detector + mesh per call)', () {
    for (final FaceMeshDelegate delegate in kDelegates) {
      for (final bool attention in kAttention) {
        if (skipConfig(delegate, attention)) {
          continue;
        }
        testWidgets('delegate=${delegate.name} attention=$attention', (
          WidgetTester tester,
        ) async {
          final FaceDetectorProcessor detector =
              await FaceDetectorProcessor.create(delegate: delegate);
          final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
            enableRoiTracking: false,
            enableSmoothing: false,
            enableAttentionMesh: attention,
            delegate: delegate,
          );

          for (final MapEntry<String, FaceMeshImage> entry
              in <String, FaceMeshImage>{
                'portrait': portrait,
              }.entries) {
            final FaceMeshImage image = entry.value;
            void runOnce() {
              final FaceDetectionResult det = detector.process(image);
              final FaceDetection? top = det.primaryDetection;
              expect(top, isNotNull, reason: 'no face detected');
              mesh.process(image, roi: top!.expandedFaceRect);
            }

            for (int i = 0; i < kWarmupRuns; i++) {
              runOnce();
            }
            final List<double> samples = <double>[];
            final Stopwatch sw = Stopwatch();
            for (int i = 0; i < kMeasuredRuns; i++) {
              sw
                ..reset()
                ..start();
              runOnce();
              sw.stop();
              samples.add(sw.elapsedMicroseconds / 1000.0);
            }
            emitResult(
              app: kApp,
              suite: 'single_image',
              config: <String, Object?>{
                'image': entry.key,
                'width': image.width,
                'height': image.height,
                'delegate': delegate.name,
                'attention': attention,
                'activeDelegate': mesh.activeDelegate.name,
              },
              samplesMs: samples,
            );
          }

          mesh.close();
          detector.close();
        });
      }
    }
  });

  group('streaming (roi tracking, detector on first frame only)', () {
    late List<FaceMeshImage> frames;

    setUpAll(() async {
      frames = await loadFrameSequence();
      expect(frames, isNotEmpty, reason: 'run tool/prepare_assets.py first');
    });

    for (final FaceMeshDelegate delegate in kDelegates) {
      for (final bool attention in kAttention) {
        if (skipConfig(delegate, attention)) {
          continue;
        }
        testWidgets('delegate=${delegate.name} attention=$attention', (
          WidgetTester tester,
        ) async {
          final FaceDetectorProcessor detector =
              await FaceDetectorProcessor.create(delegate: delegate);
          final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
            enableRoiTracking: true,
            enableAttentionMesh: attention,
            delegate: delegate,
          );

          final FaceDetectionResult det = detector.process(frames.first);
          final FaceDetection? top = det.primaryDetection;
          expect(top, isNotNull, reason: 'no face in first frame');

          final List<double> samples = <double>[];
          final List<FaceMeshResult> tracked = <FaceMeshResult>[];
          final Stopwatch sw = Stopwatch();
          for (int i = 0; i < frames.length; i++) {
            sw
              ..reset()
              ..start();
            final FaceMeshResult result = i == 0
                ? mesh.process(frames[i], roi: top!.expandedFaceRect)
                : mesh.process(frames[i]);
            sw.stop();
            if (i >= kStreamSettleFrames) {
              samples.add(sw.elapsedMicroseconds / 1000.0);
              tracked.add(result);
            }
          }

          // OneEuro landmark smoothing: applied after the fact on the same
          // results so raw and smoothed jitter come from one inference pass.
          final FaceLandmarkSmoother smoother = FaceLandmarkSmoother();
          final List<List<FaceMeshLandmark>> rawLandmarks =
              <List<FaceMeshLandmark>>[];
          final List<List<FaceMeshLandmark>> smoothedLandmarks =
              <List<FaceMeshLandmark>>[];
          double oneEuroTotalMs = 0;
          for (int i = 0; i < tracked.length; i++) {
            rawLandmarks.add(tracked[i].landmarks);
            sw
              ..reset()
              ..start();
            final FaceMeshResult smoothed = smoother.smooth(
              tracked[i],
              timestamp: Duration(milliseconds: 33 * i),
            );
            sw.stop();
            oneEuroTotalMs += sw.elapsedMicroseconds / 1000.0;
            smoothedLandmarks.add(smoothed.landmarks);
          }

          final int w = frames.first.width;
          final int h = frames.first.height;
          emitResult(
            app: kApp,
            suite: 'streaming',
            config: <String, Object?>{
              'frames': frames.length,
              'width': w,
              'height': h,
              'delegate': delegate.name,
              'attention': attention,
              'activeDelegate': mesh.activeDelegate.name,
            },
            samplesMs: samples,
            extra: <String, Object?>{
              'jitterRawPx': meanJitterPx(rawLandmarks, w, h),
              'jitterOneEuroPx': meanJitterPx(smoothedLandmarks, w, h),
              'oneEuroCostMs': oneEuroTotalMs / tracked.length,
            },
          );

          mesh.close();
          detector.close();
        });
      }
    }
  });
}
