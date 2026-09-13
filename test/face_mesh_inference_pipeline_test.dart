import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Pipeline flow tests on fake processors: tracking hand-off, input-change
/// resets, multi-face slots, and the smoothing default. Native inference is
/// covered by the bench functional suite on a device.
void main() {
  late _FakeDetector detector;
  late _FakeMesh mesh;

  FaceMeshImage frame({int width = 640, int height = 480}) => FaceMeshImage(
    pixels: Uint8List(width * height * 4),
    width: width,
    height: height,
  );

  setUp(() {
    detector = _FakeDetector();
    mesh = _FakeMesh();
  });

  group('single face', () {
    test('detector runs to acquire, then tracked frames skip it', () {
      final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
        landmarkSmoothing: null,
      );
      detector.detections = <FaceDetection>[_detection(0.5, 0.5)];

      final FaceMeshInferenceResult first = pipeline.process(frame());
      expect(first.detectorRan, isTrue);
      expect(first.meshResult, isNotNull);
      expect(pipeline.isTracking, isTrue);

      final FaceMeshInferenceResult second = pipeline.process(frame());
      expect(second.detectorRan, isFalse);
      expect(second.meshResult, isNotNull);
      expect(detector.calls, 1);
      expect(mesh.calls, 2);
    });

    test('losing the face re-acquires through the detector', () {
      final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
        landmarkSmoothing: null,
      );
      detector.detections = <FaceDetection>[_detection(0.5, 0.5)];
      pipeline.process(frame());

      mesh.nextEmpty = true;
      final FaceMeshInferenceResult lost = pipeline.process(frame());
      expect(lost.detectorRan, isTrue, reason: 'tracked call came back empty');
      expect(pipeline.isTracking, isTrue, reason: 'detector re-acquired it');
      expect(detector.calls, 2);
    });

    test('a frame size change resets tracking', () {
      final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
        landmarkSmoothing: null,
      );
      detector.detections = <FaceDetection>[_detection(0.5, 0.5)];
      pipeline.process(frame());
      pipeline.process(frame());
      expect(detector.calls, 1);

      pipeline.process(frame(width: 1280, height: 720));
      expect(detector.calls, 2);
    });

    test('runMesh false returns detections only', () {
      final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
      );
      detector.detections = <FaceDetection>[_detection(0.5, 0.5)];
      final FaceMeshInferenceResult result = pipeline.process(
        frame(),
        runMesh: false,
      );
      expect(result.detectorRan, isTrue);
      expect(result.meshResult, isNull);
      expect(mesh.calls, 0);
      expect(pipeline.isTracking, isFalse);
    });
  });

  group('multi face', () {
    test(
      'tracks keep their ids and the detector stops when slots are full',
      () {
        final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
          detector: detector,
          mesh: mesh,
          landmarkSmoothing: null,
        );
        detector.detections = <FaceDetection>[
          _detection(0.25, 0.5),
          _detection(0.75, 0.5),
        ];

        final FaceMeshMultiInferenceResult first = pipeline.processMultiFace(
          frame(),
          maxMeshFaces: 2,
        );
        expect(first.detectorRan, isTrue);
        expect(first.faces.map((TrackedFaceMesh f) => f.trackId), <int>[0, 1]);

        final FaceMeshMultiInferenceResult second = pipeline.processMultiFace(
          frame(),
          maxMeshFaces: 2,
        );
        expect(second.detectorRan, isFalse, reason: 'both slots are tracked');
        expect(second.faces.map((TrackedFaceMesh f) => f.trackId), <int>[0, 1]);
        expect(detector.calls, 1);
      },
    );

    test('a dropped track frees its slot for re-acquisition', () {
      final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
        landmarkSmoothing: null,
      );
      detector.detections = <FaceDetection>[
        _detection(0.25, 0.5),
        _detection(0.75, 0.5),
      ];
      pipeline.processMultiFace(frame(), maxMeshFaces: 2);

      // Second tracked face falls below the tracking confidence.
      mesh.scoresForNextBatch = <double>[0.9, 0.1];
      detector.detections = <FaceDetection>[_detection(0.75, 0.5)];
      final FaceMeshMultiInferenceResult result = pipeline.processMultiFace(
        frame(),
        maxMeshFaces: 2,
      );
      expect(result.detectorRan, isTrue);
      expect(result.faces.map((TrackedFaceMesh f) => f.trackId), <int>[0, 2]);
    });
  });

  group('smoothing', () {
    test('is on by default and null turns it off', () {
      expect(
        FaceMeshInferencePipeline(
          detector: detector,
          mesh: mesh,
        ).landmarkSmoothingEnabled,
        isTrue,
      );
      expect(
        FaceMeshInferencePipeline(
          detector: detector,
          mesh: mesh,
          landmarkSmoothing: null,
        ).landmarkSmoothingEnabled,
        isFalse,
      );
    });

    test('off: the mesh passes through untouched', () {
      final FaceMeshInferencePipeline pipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
        landmarkSmoothing: null,
      );
      detector.detections = <FaceDetection>[_detection(0.5, 0.5)];
      final FaceMeshInferenceResult result = pipeline.process(frame());
      expect(identical(result.meshResult, mesh.lastResult), isTrue);
    });
  });
}

FaceDetection _detection(double cx, double cy) {
  final NormalizedRect rect = NormalizedRect(
    xCenter: cx,
    yCenter: cy,
    width: 0.3,
    height: 0.4,
  );
  return FaceDetection(
    left: cx - 0.15,
    top: cy - 0.2,
    right: cx + 0.15,
    bottom: cy + 0.2,
    score: 0.9,
    faceRect: rect,
    expandedFaceRect: rect,
  );
}

/// 468 landmarks spread inside [roi] so `trackingRoi()` lands back on it.
FaceMeshResult _meshInside(
  NormalizedRect roi, {
  required int imageWidth,
  required int imageHeight,
  double score = 0.9,
}) {
  final List<FaceMeshLandmark> landmarks = List<FaceMeshLandmark>.generate(
    468,
    (int i) => FaceMeshLandmark(
      x: roi.xCenter + roi.width * ((i % 24) / 23 - 0.5) * 0.6,
      y: roi.yCenter + roi.height * ((i ~/ 24) / 19 - 0.5) * 0.6,
      z: 0,
    ),
  );
  return FaceMeshResult(
    landmarks: landmarks,
    rect: roi,
    score: score,
    imageWidth: imageWidth,
    imageHeight: imageHeight,
  );
}

class _FakeDetector implements FaceDetectorProcessor {
  List<FaceDetection> detections = <FaceDetection>[];
  int calls = 0;

  @override
  FaceDetectionResult process(
    FaceMeshFrame frame, {
    NormalizedRect? roi,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? roiScaleX,
    double? roiScaleY,
    double? roiShiftX,
    double? roiShiftY,
  }) {
    calls++;
    return FaceDetectionResult(
      detections: detections,
      imageWidth: frame.width,
      imageHeight: frame.height,
    );
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _FakeMesh implements FaceMeshProcessor {
  int calls = 0;
  bool nextEmpty = false;
  List<double>? scoresForNextBatch;
  FaceMeshResult? lastResult;
  NormalizedRect? _trackedRoi;

  @override
  bool get roiTrackingEnabled => true;

  @override
  bool get isTracking => _trackedRoi != null;

  @override
  double get minTrackingConfidence => 0.5;

  @override
  FaceMeshResult process(
    FaceMeshFrame frame, {
    NormalizedRect? roi,
    FaceMeshBox? box,
    double boxScale = 1.2,
    bool boxMakeSquare = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    calls++;
    final NormalizedRect? target = roi ?? _trackedRoi;
    if (nextEmpty || target == null) {
      nextEmpty = false;
      _trackedRoi = null;
      return lastResult = FaceMeshResult(
        landmarks: const <FaceMeshLandmark>[],
        rect: const NormalizedRect(
          xCenter: 0.5,
          yCenter: 0.5,
          width: 1,
          height: 1,
        ),
        score: 0,
        imageWidth: frame.width,
        imageHeight: frame.height,
      );
    }
    final FaceMeshResult result = _meshInside(
      target,
      imageWidth: frame.width,
      imageHeight: frame.height,
    );
    _trackedRoi = result.trackingRoi();
    return lastResult = result;
  }

  @override
  List<FaceMeshResult> processRois(
    FaceMeshFrame frame, {
    required List<NormalizedRect> rois,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) {
    calls++;
    final List<double>? scores = scoresForNextBatch;
    scoresForNextBatch = null;
    return <FaceMeshResult>[
      for (int i = 0; i < rois.length; i++)
        _meshInside(
          rois[i],
          imageWidth: frame.width,
          imageHeight: frame.height,
          score: scores != null && i < scores.length ? scores[i] : 0.9,
        ),
    ];
  }

  @override
  List<FaceMeshResult> processMultiFace(
    FaceMeshFrame frame, {
    required Iterable<FaceDetection> detections,
    int? maxMeshFaces,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) => processRois(
    frame,
    rois: <NormalizedRect>[
      for (final FaceDetection d in detections) d.expandedFaceRect,
    ],
  );

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
