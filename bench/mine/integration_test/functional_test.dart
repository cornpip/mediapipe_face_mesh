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
    faceRoi = face!.expandedFaceRect;
  });

  test('v1 with the separate iris pass returns 478 sane landmarks', () async {
    final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
      model: FaceMeshModel.v1,
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
    final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
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

      final FaceBlendshapes first = blendshapes.process(result)!;
      final FaceBlendshapes second = blendshapes.process(result)!;
      expect(first.toMap(), hasLength(FaceBlendshape.values.length));
      for (final FaceBlendshape shape in FaceBlendshape.values) {
        expect(first[shape], inInclusiveRange(0.0, 1.0));
        // The reused scratch buffer must not leak state between calls.
        expect(second[shape], closeTo(first[shape], 1e-6));
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

  test('NV21 frames run through the same entry points as RGBA', () async {
    final FaceMeshNv21Image nv21 = rgbaToNv21(portrait);
    final FaceDetectorProcessor detector = await FaceDetectorProcessor.create();
    final FaceMeshProcessor mesh = await FaceMeshProcessor.create();
    try {
      // Default model is v2 in 3.0.0.
      expect(mesh.model, FaceMeshModel.v2);
      expect(mesh.irisEnabled, isTrue);

      final FaceDetection rgbaFace = detector
          .process(portrait)
          .primaryDetection!;
      final FaceDetection nv21Face = detector.process(nv21).primaryDetection!;
      expect(
        nv21Face.faceRect.xCenter,
        closeTo(rgbaFace.faceRect.xCenter, 0.02),
      );
      expect(
        nv21Face.faceRect.yCenter,
        closeTo(rgbaFace.faceRect.yCenter, 0.02),
      );

      final FaceMeshResult rgbaMesh = mesh.process(portrait, roi: faceRoi);
      final FaceMeshResult nv21Mesh = mesh.process(nv21, roi: faceRoi);
      expect(rgbaMesh.landmarks.length, 478);
      expectLandmarksClose(nv21Mesh, rgbaMesh);

      final List<FaceMeshResult> batch = mesh.processRois(
        nv21,
        rois: <NormalizedRect>[faceRoi],
      );
      expect(batch, hasLength(1));
      expectLandmarksClose(batch.single, rgbaMesh);

      // Pipeline: smoothing on by default, NV21 and RGBA through one method.
      final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
      );
      expect(pipeline.landmarkSmoothingEnabled, isTrue);
      final FaceMeshInferenceResult single = pipeline.process(nv21);
      expect(single.meshResult, isNotNull);
      expectLandmarksClose(single.meshResult!, rgbaMesh);
      final FaceMeshMultiInferenceResult multi = pipeline.processMultiFace(
        portrait,
        maxMeshFaces: 2,
      );
      expect(multi.faces, hasLength(1));

      // Stream: the frame type is inferred from the stream.
      final FaceMeshInferenceStreamProcessor streamProcessor =
          FaceMeshInferenceStreamProcessor(pipeline);
      pipeline.resetTracking();
      final List<FaceMeshInferenceResult> streamed = await streamProcessor
          .process(
            Stream<FaceMeshNv21Image>.fromIterable(<FaceMeshNv21Image>[
              nv21,
              nv21,
            ]),
            runMeshResolver: (FaceMeshNv21Image frame) => true,
          )
          .toList();
      expect(streamed, hasLength(2));
      expect(streamed.first.detectorRan, isTrue);
      expect(streamed.last.detectorRan, isFalse, reason: 'tracked frame');
      expectLandmarksClose(streamed.last.meshResult!, rgbaMesh);
    } finally {
      mesh.close();
      detector.close();
    }
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
    int? readPlaneByte(
      Uint8List bytes,
      int rowStride,
      int pixelStride,
      int row,
      int col,
    ) {
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

/// Converts an RGBA frame to NV21 (BT.601, chroma averaged per 2x2 block).
FaceMeshNv21Image rgbaToNv21(FaceMeshImage image) {
  final int w = image.width & ~1;
  final int h = image.height & ~1;
  final Uint8List y = Uint8List(w * h);
  final Uint8List vu = Uint8List(w * (h ~/ 2));
  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col++) {
      final int i = row * image.bytesPerRow + col * 4;
      final int r = image.pixels[i];
      final int g = image.pixels[i + 1];
      final int b = image.pixels[i + 2];
      y[row * w + col] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      if (row.isEven && col.isEven) {
        final int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        final int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        final int j = (row ~/ 2) * w + col;
        vu[j] = v.clamp(0, 255);
        vu[j + 1] = u.clamp(0, 255);
      }
    }
  }
  return FaceMeshNv21Image(yPlane: y, vuPlane: vu, width: w, height: h);
}

/// Landmarks from the NV21 path may differ slightly from the RGBA path
/// because of chroma subsampling; 1% of the frame is well inside that.
void expectLandmarksClose(FaceMeshResult actual, FaceMeshResult expected) {
  expect(actual.landmarks.length, expected.landmarks.length);
  for (int i = 0; i < expected.landmarks.length; i++) {
    expect(actual.landmarks[i].x, closeTo(expected.landmarks[i].x, 0.01));
    expect(actual.landmarks[i].y, closeTo(expected.landmarks[i].y, 0.01));
  }
}
