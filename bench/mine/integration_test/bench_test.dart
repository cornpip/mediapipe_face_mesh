import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'bench_util.dart';

const String kApp = 'mediapipe_face_mesh';

/// Frames between independent detector probes in the streaming suites.
const int kDriftCheckInterval = 30;

/// Minimum IoU between the tracked landmark bbox and an independent
/// detection before the run is considered drifted. The two boxes have
/// different tightness, so healthy tracking sits well above this floor
/// while a frozen or wandering ROI collapses toward zero.
const double kDriftIouFloor = 0.2;

/// Delegate/model matrix. cpu runs every model; xnnpack runs v1 (cpu
/// parity) and v2 (checks full-graph delegation) but skips attention
/// (custom ops split its graph, so it mirrors cpu). gpuV2 is deprecated
/// and excluded.
const List<FaceMeshDelegate> kDelegates = <FaceMeshDelegate>[
  FaceMeshDelegate.cpu,
  FaceMeshDelegate.xnnpack,
];
const List<FaceMeshModel> kModels = <FaceMeshModel>[
  FaceMeshModel.v1,
  FaceMeshModel.attention,
  FaceMeshModel.v2,
];

bool skipConfig(FaceMeshDelegate delegate, FaceMeshModel model) =>
    delegate == FaceMeshDelegate.xnnpack && model == FaceMeshModel.attention;

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

Future<List<String>> listFramePaths() async {
  final AssetManifest manifest = await AssetManifest.loadFromAssetBundle(
    rootBundle,
  );
  return manifest
      .listAssets()
      .where(
        (String p) => p.startsWith('assets/frames/') && p.endsWith('.jpg'),
      )
      .toList()
    ..sort();
}

/// Source-video fps recorded by tool/prepare_assets.py.
Future<double> loadFrameFps() async {
  String? raw;
  try {
    raw = await rootBundle.loadString('assets/frames/meta.json');
  } catch (_) {
    // Missing asset; reported through the expect below.
  }
  expect(
    raw,
    isNotNull,
    reason: 'assets/frames/meta.json missing; re-run tool/prepare_assets.py',
  );
  final Map<String, Object?> meta =
      jsonDecode(raw!) as Map<String, Object?>;
  return (meta['fps']! as num).toDouble();
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

/// IoU between the landmarks' axis-aligned bbox and a detection box, both in
/// normalized coordinates.
double landmarkDetectionIou(
  List<FaceMeshLandmark> landmarks,
  FaceDetection detection,
) {
  double minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (final FaceMeshLandmark lm in landmarks) {
    minX = math.min(minX, lm.x);
    minY = math.min(minY, lm.y);
    maxX = math.max(maxX, lm.x);
    maxY = math.max(maxY, lm.y);
  }
  final double ix =
      math.min(maxX, detection.right) - math.max(minX, detection.left);
  final double iy =
      math.min(maxY, detection.bottom) - math.max(minY, detection.top);
  if (ix <= 0 || iy <= 0) {
    return 0;
  }
  final double inter = ix * iy;
  final double union =
      (maxX - minX) * (maxY - minY) +
      (detection.right - detection.left) * (detection.bottom - detection.top) -
      inter;
  return union <= 0 ? 0 : inter / union;
}

/// Runs one streaming config: detector on frame 0 only, ROI tracking after.
///
/// The default (back-to-back) variant preloads all decoded frames so the
/// measured loop touches nothing but the inference call (the original
/// hot-loop protocol; per-frame decode between calls pollutes caches and
/// roughly doubles the measured latency). With [pacedFps] set, each frame
/// is instead decoded in the loop (outside the stopwatch, standing in for
/// camera frame delivery) and waits for its wall-clock deadline at that
/// cadence. Idle gaps let the CPU governor drop clocks, so a paced run
/// measures device DVFS behavior as much as the package; treat it as an
/// exploratory scenario test, not a tracked metric.
Future<void> runStreamingBench({
  required FaceMeshDelegate delegate,
  required FaceMeshModel model,
  double? pacedFps,
}) async {
  await thermalCooldown();

  final List<String> framePaths = await listFramePaths();
  expect(framePaths, isNotEmpty, reason: 'run tool/prepare_assets.py first');
  final double fps = await loadFrameFps();
  final Duration frameInterval = Duration(
    microseconds: (1e6 / fps).round(),
  );
  final bool paced = pacedFps != null;
  final Duration? paceInterval = pacedFps == null
      ? null
      : Duration(microseconds: (1e6 / pacedFps).round());

  final FaceDetectorProcessor detector = await FaceDetectorProcessor.create(
    delegate: delegate,
  );
  final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
    enableRoiTracking: true,
    model: model,
    delegate: delegate,
  );
  final int expectedLandmarks = model == FaceMeshModel.v1 ? 468 : 478;

  final List<double> samples = <double>[];
  final List<FaceMeshResult> tracked = <FaceMeshResult>[];
  final List<double> driftIous = <double>[];
  int trackingFailFrames = 0;
  int noFaceFrames = 0;
  double scoreMin = double.infinity;
  int width = 0;
  int height = 0;
  NormalizedRect? firstRoi;

  List<FaceMeshImage>? preloaded;
  if (!paced) {
    preloaded = <FaceMeshImage>[];
    for (final String p in framePaths) {
      preloaded.add(await loadRgbaAsset(p));
    }
  }

  final Stopwatch sw = Stopwatch();
  final Stopwatch wall = Stopwatch();
  for (int i = 0; i < framePaths.length; i++) {
    final FaceMeshImage frame = paced
        ? await loadRgbaAsset(framePaths[i])
        : preloaded![i];
    width = frame.width;
    height = frame.height;
    if (i == 0) {
      final FaceDetectionResult det = detector.process(frame);
      final FaceDetection? top = det.primaryDetection;
      expect(top, isNotNull, reason: 'no face in first frame');
      firstRoi = top!.expandedFaceRect;
      wall.start();
    }
    if (paceInterval != null) {
      final Duration ahead = paceInterval * i - wall.elapsed;
      if (ahead > Duration.zero) {
        await Future<void>.delayed(ahead);
      }
    }
    sw
      ..reset()
      ..start();
    final FaceMeshResult result = i == 0
        ? mesh.process(frame, roi: firstRoi)
        : mesh.process(frame);
    sw.stop();

    final bool meshValid =
        result.landmarks.length == expectedLandmarks && result.score > 0.5;
    if (meshValid) {
      scoreMin = math.min(scoreMin, result.score);
    } else if (detector.process(frame).primaryDetection != null) {
      // Independent detection outside the stopwatch still finds a face, so
      // the invalid mesh output is a tracking failure, not a no-face frame
      // (the source video fades to black over its last frames).
      trackingFailFrames++;
    } else {
      noFaceFrames++;
    }
    if (i >= kStreamSettleFrames) {
      samples.add(sw.elapsedMicroseconds / 1000.0);
      tracked.add(result);
    }
    if (meshValid && i > 0 && i % kDriftCheckInterval == 0) {
      // Drift probe: an independent detector pass outside the stopwatch. A
      // frozen or wandering tracker still yields low jitter and normal
      // latency, so only a fresh detection can expose it.
      final FaceDetection? probe = detector.process(frame).primaryDetection;
      if (probe != null) {
        driftIous.add(landmarkDetectionIou(result.landmarks, probe));
      }
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
      timestamp: frameInterval * i,
    );
    sw.stop();
    oneEuroTotalMs += sw.elapsedMicroseconds / 1000.0;
    smoothedLandmarks.add(smoothed.landmarks);
  }

  emitResult(
    app: kApp,
    suite: paced ? 'streaming_paced' : 'streaming',
    config: <String, Object?>{
      'frames': framePaths.length,
      'width': width,
      'height': height,
      'fps': fps,
      'pacedFps': ?pacedFps,
      'delegate': delegate.name,
      'model': model.name,
      'activeDelegate': mesh.activeDelegate.name,
    },
    samplesMs: samples,
    extra: <String, Object?>{
      'jitterRawPx': meanJitterPx(rawLandmarks, width, height),
      'jitterOneEuroPx': meanJitterPx(smoothedLandmarks, width, height),
      'oneEuroCostMs': oneEuroTotalMs / tracked.length,
      'trackingFailFrames': trackingFailFrames,
      'noFaceFrames': noFaceFrames,
      if (scoreMin.isFinite) 'scoreMin': scoreMin,
      if (driftIous.isNotEmpty)
        'roiDriftIouMean':
            driftIous.reduce((double a, double b) => a + b) /
            driftIous.length,
      if (driftIous.isNotEmpty) 'roiDriftIouMin': driftIous.reduce(math.min),
    },
  );

  mesh.close();
  detector.close();

  // Asserted after emitResult so the numbers still print on failure.
  expect(
    trackingFailFrames,
    0,
    reason: 'mesh output invalid while an independent detection finds a face',
  );
  expect(driftIous, isNotEmpty, reason: 'no successful drift probes');
  expect(
    driftIous.reduce(math.min),
    greaterThan(kDriftIouFloor),
    reason: 'tracked landmarks drifted away from an independent detection',
  );
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late FaceMeshImage portrait;

  setUpAll(() async {
    portrait = await loadRgbaAsset('assets/portrait.jpg');
  });

  group('single image (cold pipeline: detector + mesh per call)', () {
    for (final FaceMeshDelegate delegate in kDelegates) {
      for (final FaceMeshModel model in kModels) {
        if (skipConfig(delegate, model)) {
          continue;
        }
        testWidgets('delegate=${delegate.name} model=${model.name}', (
          WidgetTester tester,
        ) async {
          await thermalCooldown();

          final FaceDetectorProcessor detector =
              await FaceDetectorProcessor.create(delegate: delegate);
          final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
            enableRoiTracking: false,
            enableSmoothing: false,
            model: model,
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
                'model': model.name,
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

  void streamingMatrix({double? pacedFps}) {
    for (final FaceMeshDelegate delegate in kDelegates) {
      for (final FaceMeshModel model in kModels) {
        if (skipConfig(delegate, model)) {
          continue;
        }
        testWidgets('delegate=${delegate.name} model=${model.name}', (
          WidgetTester tester,
        ) async {
          await runStreamingBench(
            delegate: delegate,
            model: model,
            pacedFps: pacedFps,
          );
        });
      }
    }
  }

  group('streaming (back-to-back, roi tracking)', streamingMatrix);

  // Opt-in camera-cadence scenario: pass --dart-define=PACED_FPS=<fps> to
  // also run the streaming matrix paced at that cadence.
  const String pacedFpsEnv = String.fromEnvironment('PACED_FPS');
  if (pacedFpsEnv.isNotEmpty) {
    final double? pacedFps = double.tryParse(pacedFpsEnv);
    if (pacedFps == null || pacedFps <= 0) {
      throw ArgumentError('invalid PACED_FPS: $pacedFpsEnv');
    }
    group('streaming_paced (${pacedFpsEnv}fps cadence, roi tracking)', () {
      streamingMatrix(pacedFps: pacedFps);
    });
  }
}
