import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

Duration _frameTime(int index, {int fps = 30}) =>
    Duration(microseconds: 1 + index * 1000000 ~/ fps);

double _std(List<double> values) {
  final double mean = values.reduce((double a, double b) => a + b) /
      values.length;
  double sum = 0;
  for (final double value in values) {
    sum += (value - mean) * (value - mean);
  }
  return math.sqrt(sum / values.length);
}

void main() {
  group('OneEuroFilter', () {
    test('first sample passes through unfiltered', () {
      final OneEuroFilter filter = OneEuroFilter(minCutoff: 0.05, beta: 80);
      expect(filter.apply(_frameTime(0), 5.0), 5.0);
    });

    test('non-increasing timestamp returns value unfiltered', () {
      final OneEuroFilter filter = OneEuroFilter(minCutoff: 0.05, beta: 80);
      filter.apply(_frameTime(0), 1.0);
      filter.apply(_frameTime(1), 2.0);
      expect(filter.apply(_frameTime(1), 100.0), 100.0);
      // The rejected sample must not have advanced filter state.
      final double next = filter.apply(_frameTime(2), 2.0);
      expect((next - 2.0).abs(), lessThan(0.5));
    });

    test('attenuates jitter on a static signal', () {
      // beta 0 isolates the fixed-cutoff smoothing behavior; at 30 fps and
      // minCutoff 1 Hz the filter is an EMA with alpha ~0.17, which cuts a
      // white-noise std roughly threefold.
      final OneEuroFilter filter = OneEuroFilter(minCutoff: 1.0, beta: 0.0);
      final math.Random random = math.Random(7);
      final List<double> raw = <double>[];
      final List<double> smoothed = <double>[];
      for (int i = 0; i < 120; i++) {
        final double value = 100.0 + (random.nextDouble() - 0.5) * 2.0;
        raw.add(value);
        smoothed.add(filter.apply(_frameTime(i), value));
      }
      // Skip the settle-in phase before measuring.
      final double rawStd = _std(raw.sublist(30));
      final double smoothedStd = _std(smoothed.sublist(30));
      expect(smoothedStd, lessThan(rawStd / 2));
    });

    test('follows fast motion with little lag', () {
      final OneEuroFilter filter = OneEuroFilter(minCutoff: 0.05, beta: 80);
      double value = 0;
      double smoothed = 0;
      for (int i = 0; i < 60; i++) {
        value = i * 10.0; // 300 units/second at 30 fps.
        smoothed = filter.apply(_frameTime(i), value);
      }
      // With beta velocity adaptation the lag stays under one frame step.
      expect((value - smoothed).abs(), lessThan(10.0));
    });
  });

  group('OneEuroLandmarksSmoother', () {
    List<SmoothablePoint> jitteredFace(math.Random random, double amplitude) =>
        <SmoothablePoint>[
          for (int i = 0; i < 4; i++)
            (
              x: 0.4 + 0.05 * i + (random.nextDouble() - 0.5) * amplitude,
              y: 0.4 + 0.05 * i + (random.nextDouble() - 0.5) * amplitude,
              z: -0.01 * i,
            ),
        ];

    test('reduces per-landmark jitter on a static face', () {
      final OneEuroLandmarksSmoother smoother = OneEuroLandmarksSmoother();
      final math.Random random = math.Random(11);
      final List<double> rawX = <double>[];
      final List<double> smoothedX = <double>[];
      for (int i = 0; i < 120; i++) {
        // Sub-pixel jitter, like real capture noise on a still face.
        final List<SmoothablePoint> face = jitteredFace(random, 0.001);
        final List<SmoothablePoint> result = smoother.apply(
          landmarks: face,
          imageWidth: 640,
          imageHeight: 480,
          timestamp: _frameTime(i),
        );
        rawX.add(face[0].x);
        smoothedX.add(result[0].x);
      }
      expect(
        _std(smoothedX.sublist(30)),
        lessThan(_std(rawX.sublist(30)) * 0.6),
      );
    });

    test('empty input and degenerate scale pass through', () {
      final OneEuroLandmarksSmoother smoother = OneEuroLandmarksSmoother();
      expect(
        smoother.apply(
          landmarks: const <SmoothablePoint>[],
          imageWidth: 640,
          imageHeight: 480,
          timestamp: _frameTime(0),
        ),
        isEmpty,
      );
      const List<SmoothablePoint> collapsed = <SmoothablePoint>[
        (x: 0.5, y: 0.5, z: 0),
        (x: 0.5, y: 0.5, z: 0),
      ];
      final List<SmoothablePoint> result = smoother.apply(
        landmarks: collapsed,
        imageWidth: 640,
        imageHeight: 480,
        timestamp: _frameTime(0),
      );
      expect(result, same(collapsed));
    });

    test('landmark count change resets filter state', () {
      final OneEuroLandmarksSmoother smoother = OneEuroLandmarksSmoother();
      final math.Random random = math.Random(3);
      for (int i = 0; i < 10; i++) {
        smoother.apply(
          landmarks: jitteredFace(random, 0.004),
          imageWidth: 640,
          imageHeight: 480,
          timestamp: _frameTime(i),
        );
      }
      // A different landmark count must restart smoothing: the first frame
      // of the new layout passes through unchanged.
      final List<SmoothablePoint> bigger = <SmoothablePoint>[
        ...jitteredFace(random, 0),
        (x: 0.9, y: 0.9, z: 0),
      ];
      final List<SmoothablePoint> result = smoother.apply(
        landmarks: bigger,
        imageWidth: 640,
        imageHeight: 480,
        timestamp: _frameTime(10),
      );
      for (int i = 0; i < bigger.length; i++) {
        expect(result[i].x, closeTo(bigger[i].x, 1e-9));
        expect(result[i].y, closeTo(bigger[i].y, 1e-9));
      }
    });

    test('velocity normalization makes small and large faces behave alike',
        () {
      // The same face pattern rendered at 1x and 4x scale, with jitter
      // proportional to the face size, must smooth to proportional outputs
      // when value scaling is enabled.
      final OneEuroLandmarksSmoother small = OneEuroLandmarksSmoother();
      final OneEuroLandmarksSmoother large = OneEuroLandmarksSmoother();
      final math.Random random = math.Random(5);
      double smallLast = 0;
      double largeLast = 0;
      for (int i = 0; i < 60; i++) {
        final double jitter = (random.nextDouble() - 0.5) * 0.004;
        final List<SmoothablePoint> base = <SmoothablePoint>[
          (x: 0.10 + jitter, y: 0.10, z: 0),
          (x: 0.15, y: 0.15, z: 0),
        ];
        final List<SmoothablePoint> scaled = <SmoothablePoint>[
          for (final SmoothablePoint p in base)
            (x: p.x * 4, y: p.y * 4, z: p.z),
        ];
        smallLast = small
            .apply(
              landmarks: base,
              imageWidth: 640,
              imageHeight: 480,
              timestamp: _frameTime(i),
            )[0]
            .x;
        largeLast = large
            .apply(
              landmarks: scaled,
              imageWidth: 640,
              imageHeight: 480,
              timestamp: _frameTime(i),
            )[0]
            .x;
      }
      expect(largeLast, closeTo(smallLast * 4, 1e-6));
    });
  });

  group('FaceLandmarkSmoother', () {
    FaceMeshResult resultWith(List<FaceMeshLandmark> landmarks) =>
        FaceMeshResult(
          landmarks: landmarks,
          rect: const NormalizedRect(
            xCenter: 0.5,
            yCenter: 0.5,
            width: 0.5,
            height: 0.5,
          ),
          score: 0.9,
          imageWidth: 640,
          imageHeight: 480,
        );

    test('keeps rect and score, smooths landmarks', () {
      final FaceLandmarkSmoother smoother = FaceLandmarkSmoother();
      final math.Random random = math.Random(13);
      FaceMeshResult? smoothed;
      final List<double> rawX = <double>[];
      final List<double> smoothedX = <double>[];
      for (int i = 0; i < 120; i++) {
        final FaceMeshResult raw = resultWith(<FaceMeshLandmark>[
          FaceMeshLandmark(
            x: 0.4 + (random.nextDouble() - 0.5) * 0.001,
            y: 0.4,
            z: -0.01,
          ),
          FaceMeshLandmark(x: 0.6, y: 0.6, z: -0.02),
        ]);
        smoothed = smoother.smooth(raw, timestamp: _frameTime(i));
        rawX.add(raw.landmarks[0].x);
        smoothedX.add(smoothed.landmarks[0].x);
      }
      expect(smoothed!.score, 0.9);
      expect(smoothed.rect.width, 0.5);
      expect(
        _std(smoothedX.sublist(30)),
        lessThan(_std(rawX.sublist(30)) * 0.6),
      );
    });

    test('image size change resets state', () {
      final FaceLandmarkSmoother smoother = FaceLandmarkSmoother();
      for (int i = 0; i < 10; i++) {
        smoother.smooth(
          resultWith(<FaceMeshLandmark>[
            FaceMeshLandmark(x: 0.2, y: 0.2, z: 0),
            FaceMeshLandmark(x: 0.4, y: 0.4, z: 0),
          ]),
          timestamp: _frameTime(i),
        );
      }
      final FaceMeshResult other = FaceMeshResult(
        landmarks: <FaceMeshLandmark>[
          FaceMeshLandmark(x: 0.8, y: 0.8, z: 0),
          FaceMeshLandmark(x: 0.9, y: 0.9, z: 0),
        ],
        rect: const NormalizedRect(
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1,
          height: 1,
        ),
        score: 1,
        imageWidth: 1280,
        imageHeight: 720,
      );
      final FaceMeshResult smoothed = smoother.smooth(
        other,
        timestamp: _frameTime(10),
      );
      // Fresh sequence: the first frame passes through unchanged.
      expect(smoothed.landmarks[0].x, closeTo(0.8, 1e-9));
      expect(smoothed.landmarks[1].y, closeTo(0.9, 1e-9));
    });
  });
}
