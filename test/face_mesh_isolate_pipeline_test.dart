import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Fake processors built inside the worker cover the round trip, tracking
/// state, error transport, and shutdown. Native inference is covered on
/// device by the bench suite.
void main() {
  FaceMeshImage frame({int width = 640, int height = 480}) => FaceMeshImage(
    pixels: Uint8List(width * height * 4),
    width: width,
    height: height,
  );

  test('keeps tracking state in the worker across calls', () async {
    final FaceMeshIsolatePipeline pipeline =
        await FaceMeshIsolatePipeline.spawn(createFakePipeline, 0.5);
    addTearDown(pipeline.close);

    expect(pipeline.isBusy, isFalse);
    final Future<FaceMeshInferenceResult> pending = pipeline.process(frame());
    expect(pipeline.isBusy, isTrue);
    final FaceMeshInferenceResult first = await pending;
    expect(pipeline.isBusy, isFalse);
    expect(first.detectorRan, isTrue);
    expect(first.meshResult, isNotNull);
    expect(first.meshResult!.landmarks, hasLength(468));

    final FaceMeshInferenceResult second = await pipeline.process(frame());
    expect(second.detectorRan, isFalse);
    expect(second.meshResult, isNotNull);

    await pipeline.resetTracking();
    final FaceMeshInferenceResult third = await pipeline.process(frame());
    expect(third.detectorRan, isTrue);
  });

  test('runs the multi-face flow and forwards ArgumentError', () async {
    final FaceMeshIsolatePipeline pipeline =
        await FaceMeshIsolatePipeline.spawn(createFakePipeline, 0.5);
    addTearDown(pipeline.close);

    final FaceMeshMultiInferenceResult result = await pipeline.processMultiFace(
      frame(),
      maxMeshFaces: 2,
    );
    expect(result.faces, hasLength(1));
    expect(result.faces.single.mesh.landmarks, hasLength(468));

    await expectLater(
      pipeline.processMultiFace(frame(), maxMeshFaces: 0),
      throwsArgumentError,
    );
    // The worker survives a failed request.
    final FaceMeshMultiInferenceResult again = await pipeline.processMultiFace(
      frame(),
      maxMeshFaces: 2,
    );
    expect(again.faces, hasLength(1));
  });

  test('forwards a processor failure as the thrown exception', () async {
    final FaceMeshIsolatePipeline pipeline =
        await FaceMeshIsolatePipeline.spawn(
          createFailingPipeline,
          'detector failed',
        );
    addTearDown(pipeline.close);

    await expectLater(
      pipeline.process(frame()),
      throwsA(
        isA<FaceMeshException>().having(
          (e) => e.message,
          'message',
          'detector failed',
        ),
      ),
    );
  });

  test('spawn rejects a factory that captures an object', () async {
    final _Holder holder = _Holder();
    await expectLater(
      FaceMeshIsolatePipeline.spawn(holder.createPipeline, null),
      throwsA(
        isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains('top-level or static function'),
        ),
      ),
    );
  });

  test('spawn rethrows a factory failure', () async {
    await expectLater(
      FaceMeshIsolatePipeline.spawn(throwingFactory, null),
      throwsA(
        isA<FaceMeshException>().having(
          (e) => e.message,
          'message',
          'factory failed',
        ),
      ),
    );
  });

  test('close is idempotent and rejects later calls', () async {
    final FaceMeshIsolatePipeline pipeline =
        await FaceMeshIsolatePipeline.spawn(createFakePipeline, 0.5);
    await pipeline.process(frame());
    await pipeline.close();
    expect(pipeline.isClosed, isTrue);
    await pipeline.close();
    expect(() => pipeline.process(frame()), throwsStateError);
  });
}

/// The argument places the fake detection, which shows it reached the worker.
FaceMeshInferencePipeline createFakePipeline(double center) =>
    FaceMeshInferencePipeline(
      detector: FakeDetector(
        detections: <FaceDetection>[detection(center, center)],
      ),
      mesh: FakeMesh(),
      landmarkSmoothing: null,
    );

FaceMeshInferencePipeline createFailingPipeline(String message) =>
    FaceMeshInferencePipeline(
      detector: FakeDetector(failWith: message),
      mesh: FakeMesh(),
      landmarkSmoothing: null,
    );

FaceMeshInferencePipeline throwingFactory(Object? _) {
  throw FaceMeshException('factory failed');
}

FaceDetection detection(double cx, double cy) {
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

FaceMeshResult meshInside(
  NormalizedRect roi, {
  required int imageWidth,
  required int imageHeight,
}) {
  final double left = roi.xCenter - roi.width / 2;
  final double top = roi.yCenter - roi.height / 2;
  return FaceMeshResult(
    landmarks: <FaceMeshLandmark>[
      for (int i = 0; i < 468; i++)
        FaceMeshLandmark(
          x: left + roi.width * (0.2 + 0.6 * (i % 10) / 9),
          y: top + roi.height * (0.2 + 0.6 * (i ~/ 10) / 47),
          z: 0,
        ),
    ],
    rect: roi,
    score: 0.9,
    imageWidth: imageWidth,
    imageHeight: imageHeight,
  );
}

/// Holds something unsendable so an instance-method tear-off cannot cross
/// the isolate boundary.
class _Holder {
  final ReceivePort port = ReceivePort();

  FaceMeshInferencePipeline createPipeline(Object? _) =>
      createFakePipeline(0.5);
}

class FakeDetector implements FaceDetectorProcessor {
  FakeDetector({this.detections = const <FaceDetection>[], this.failWith});

  final List<FaceDetection> detections;
  final String? failWith;

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
    if (failWith != null) {
      throw FaceMeshException(failWith!);
    }
    return FaceDetectionResult(
      detections: detections,
      imageWidth: frame.width,
      imageHeight: frame.height,
    );
  }

  @override
  void close() {}

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class FakeMesh implements FaceMeshProcessor {
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
  }) => meshInside(roi!, imageWidth: frame.width, imageHeight: frame.height);

  @override
  List<FaceMeshResult> processRois(
    FaceMeshFrame frame, {
    required List<NormalizedRect> rois,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
  }) => <FaceMeshResult>[
    for (final NormalizedRect roi in rois)
      meshInside(roi, imageWidth: frame.width, imageHeight: frame.height),
  ];

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
  void close() {}

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
