import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'bench_test.dart' show loadRgbaAsset;

/// Functional exercises for native paths the bench matrix does not cover:
/// the separate iris pass, the multi-ROI batch, the geometry and blendshapes
/// post-processors, creation option validation, and the NV21 plane
/// conversion cost.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late FaceMeshImage portrait;
  late NormalizedRect faceRoi;

  setUpAll(() async {
    portrait = await loadRgbaAsset('assets/portrait.jpg');
    final FaceDetectorProcessor detector = await FaceDetectorProcessor.create();
    final FaceDetectionResult detections = detector.process(portrait);
    detector.close();
    final FaceDetection? face = detections.primaryDetection;
    expect(face, isNotNull, reason: 'portrait.jpg must contain a face');
    faceRoi = (face!.expandedFaceRect ?? face.faceRect)!;
  });

  test('v1 with the separate iris pass returns 478 sane landmarks', () async {
    final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
      enableIris: true,
    );
    try {
      final FaceMeshResult result = mesh.process(portrait, roi: faceRoi);
      expect(result.landmarks, hasLength(478));
      expect(result.score, greaterThan(0.5));
      // Iris landmarks must land inside the face ROI's neighborhood; a
      // broken pixel/normalized decision throws them across the frame.
      for (int i = 468; i < 478; i++) {
        final FaceMeshLandmark landmark = result.landmarks[i];
        expect(
          (landmark.x - faceRoi.xCenter).abs(),
          lessThan(faceRoi.width),
          reason: 'iris landmark $i x',
        );
        expect(
          (landmark.y - faceRoi.yCenter).abs(),
          lessThan(faceRoi.height),
          reason: 'iris landmark $i y',
        );
      }
    } finally {
      mesh.close();
    }
  });

  test('multi-ROI batch keeps per-ROI results independent', () async {
    final FaceMeshProcessor mesh = await FaceMeshProcessor.createForMultiFace(
      model: FaceMeshModel.v2,
    );
    try {
      // One real face ROI plus one background corner ROI: the batch must
      // return both entries in order, the face with landmarks and the
      // corner without, instead of failing as a whole.
      const NormalizedRect cornerRoi = NormalizedRect(
        xCenter: 0.03,
        yCenter: 0.03,
        width: 0.05,
        height: 0.05,
      );
      final List<FaceMeshResult> results = mesh.processRois(
        portrait,
        rois: <NormalizedRect>[faceRoi, cornerRoi],
      );
      expect(results, hasLength(2));
      expect(results[0].landmarks, hasLength(478));
      expect(results[0].score, greaterThan(0.5));
      expect(results[1].landmarks, isEmpty);
    } finally {
      mesh.close();
    }
  });

  test('geometry and blendshapes stay sane and repeatable', () async {
    final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
      model: FaceMeshModel.v2,
    );
    final FaceBlendshapesProcessor blendshapes =
        await FaceBlendshapesProcessor.create();
    try {
      final FaceMeshResult result = mesh.process(portrait, roi: faceRoi);
      expect(result.landmarks, hasLength(478));

      final FaceMeshGeometry geometry = result.estimateGeometry();
      final FaceMeshMeasurements measurements = geometry.measurements;
      expect(measurements.interpupillaryDistance, isNotNull);
      expect(
        measurements.interpupillaryDistance!.valueCm,
        inInclusiveRange(4.0, 9.0),
      );
      expect(measurements.faceWidth.valueCm, inInclusiveRange(8.0, 22.0));
      expect(geometry.headPose.yawDegrees.isFinite, isTrue);

      final Map<FaceBlendshape, double> first = blendshapes.process(result)!;
      final Map<FaceBlendshape, double> second = blendshapes.process(result)!;
      expect(first, hasLength(FaceBlendshape.values.length));
      for (final FaceBlendshape shape in FaceBlendshape.values) {
        expect(first[shape], inInclusiveRange(0.0, 1.0));
        // The reused scratch buffer must not leak state between calls.
        expect(second[shape], closeTo(first[shape]!, 1e-6));
      }
    } finally {
      blendshapes.close();
      mesh.close();
    }
  });

  test('out-of-range creation options throw ArgumentError', () async {
    await expectLater(
      FaceMeshProcessor.create(minTrackingConfidence: 1.5),
      throwsArgumentError,
    );
    await expectLater(
      FaceMeshProcessor.create(minFacePresenceConfidence: -0.1),
      throwsArgumentError,
    );
    await expectLater(
      FaceDetectorProcessor.create(maxResults: 0),
      throwsArgumentError,
    );
    await expectLater(
      FaceDetectorProcessor.create(threads: 0),
      throwsArgumentError,
    );
  });

  test('NV21 chroma conversion timing', () {
    const int width = 1280;
    const int height = 720;
    const int rowStride = 1280;
    final Uint8List y = Uint8List(rowStride * height);
    // Camera semi-planar layout: U and V strided by 2.
    final Uint8List u = Uint8List(rowStride * (height ~/ 2));
    final Uint8List v = Uint8List(rowStride * (height ~/ 2));
    for (int i = 0; i < u.length; i++) {
      u[i] = i & 0xff;
      v[i] = (i * 7) & 0xff;
    }

    FaceMeshNv21Image? convert() => FaceMeshNv21Image.tryFromYuv420Planes(
      width: width,
      height: height,
      yPlane: FaceMeshImagePlane(bytes: y, bytesPerRow: rowStride),
      uPlane: FaceMeshImagePlane(
        bytes: u,
        bytesPerRow: rowStride,
        bytesPerPixel: 2,
      ),
      vPlane: FaceMeshImagePlane(
        bytes: v,
        bytesPerRow: rowStride,
        bytesPerPixel: 2,
      ),
    );

    // Warmup + correctness spot check.
    final FaceMeshNv21Image image = convert()!;
    expect(image.vuPlane[0], v[0]);
    expect(image.vuPlane[1], u[0]);
    expect(image.vuPlane[2], v[2]);

    const int runs = 100;
    final Stopwatch clock = Stopwatch()..start();
    for (int i = 0; i < runs; i++) {
      convert();
    }
    clock.stop();

    // The pre-2.9.0 conversion (per-pixel nullable reads), replicated here
    // as the comparison baseline.
    int? readPlaneByte(Uint8List bytes, int rowStride, int pixelStride,
        int row, int col) {
      final int index = row * rowStride + col * pixelStride;
      if (index < 0 || index >= bytes.length) {
        return null;
      }
      return bytes[index];
    }

    Uint8List? oldConvert() {
      final Uint8List out = Uint8List(rowStride * height);
      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          final int? value = readPlaneByte(y, rowStride, 1, row, col);
          if (value == null) return null;
          out[row * width + col] = value;
        }
      }
      final int uvWidth = width ~/ 2;
      final int uvHeight = height ~/ 2;
      final Uint8List vu = Uint8List(width * uvHeight);
      for (int row = 0; row < uvHeight; row++) {
        for (int col = 0; col < uvWidth; col++) {
          final int? uValue = readPlaneByte(u, rowStride, 2, row, col);
          final int? vValue = readPlaneByte(v, rowStride, 2, row, col);
          if (uValue == null || vValue == null) return null;
          final int out = row * width + col * 2;
          vu[out] = vValue;
          vu[out + 1] = uValue;
        }
      }
      return vu;
    }

    oldConvert(); // warmup
    final Stopwatch oldClock = Stopwatch()..start();
    for (int i = 0; i < runs; i++) {
      oldConvert();
    }
    oldClock.stop();

    // ignore: avoid_print
    print(
      'FUNC_BENCH tryFromYuv420Planes 720p semi-planar: '
      '${(clock.elapsedMicroseconds / runs / 1000).toStringAsFixed(3)} ms/frame '
      '(pre-2.9.0 loop: '
      '${(oldClock.elapsedMicroseconds / runs / 1000).toStringAsFixed(3)} ms/frame)',
    );
  });
}
