import 'package:flutter/foundation.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Face mesh model selection: base mesh, base + iris two-pass, or one of the
/// unified 478-landmark models (attention, FaceMesh-V2). A single choice
/// avoids ambiguous combinations.
enum MeshMode {
  base('Mesh (468)'),
  iris('Mesh (468) + Iris (10)'),
  attention('Attention Mesh (478)'),
  faceMeshV2('FaceMesh-V2 (478, upstream)');

  const MeshMode(this.label);

  final String label;

  FaceMeshModel get model => switch (this) {
    MeshMode.base || MeshMode.iris => FaceMeshModel.v1,
    MeshMode.attention => FaceMeshModel.attention,
    MeshMode.faceMeshV2 => FaceMeshModel.v2,
  };

  bool get enableIris => this == MeshMode.iris;
}

/// Model choices the worker isolate builds its pipeline from. A record of
/// enums, so it is sendable as is.
typedef IsolatePipelineArgs = ({
  FaceDetectionModel detectionModel,
  MeshMode meshMode,
});

/// XNNPACK with the default CPU fallback: same speed as cpu on Android,
/// 4~5x faster on Windows.
const FaceMeshDelegate preferredDelegate = FaceMeshDelegate.xnnpack;

/// Faces tracked at once in the multi-face flow, and the detector's
/// candidate count.
const int maxMeshFaces = 4;

/// OneEuro landmark smoothing (official FaceLandmarker stream-mode
/// behavior): removes per-point jitter on a still face while fast head
/// motion passes through with almost no lag. The demo always enables it.
const LandmarkSmoothingOptions landmarkSmoothing = LandmarkSmoothingOptions();

/// Top-level so the worker isolate builds its processors with the same
/// options (see [createPipelineInWorker]).
Future<FaceDetectorProcessor> createFaceDetectorProcessor(
  FaceDetectionModel model,
) {
  final isFullRange = model != FaceDetectionModel.shortRange;
  return FaceDetectorProcessor.create(
    model: model,
    delegate: preferredDelegate,
    // Let the detector return several candidates; the single-face flow
    // still picks the best one, and the multi-face flow needs them all.
    maxResults: maxMeshFaces,
    // Detector ROI defaults are scaleX/scaleY = 1.5 and shiftX/shiftY = 0.0.
    // This demo keeps the default X values and only nudges Y; with landmark
    // tracking these apply to (re)acquisition frames only. Tune per
    // model/camera if the acquisition box is too loose or tight.
    roiScaleY: isFullRange ? 1.6 : 1.7,
    roiShiftY: isFullRange ? -0.1 : -0.2,
  );
}

Future<FaceMeshProcessor> createFaceMeshProcessor({
  required FaceMeshModel model,
  required bool iris,
}) async {
  final FaceMeshProcessor processor = await FaceMeshProcessor.create(
    model: model,
    enableIris: iris,
    delegate: preferredDelegate,
  );
  debugPrint(
    'FaceMeshProcessor created: model=$model iris=$iris '
    'delegate=${processor.activeDelegate}',
  );
  return processor;
}

/// Runs inside the worker isolate. Top-level, so it can be handed to
/// FaceMeshIsolatePipeline.spawn without capturing a State.
Future<FaceMeshInferencePipeline> createPipelineInWorker(
  IsolatePipelineArgs args,
) async => FaceMeshInferencePipeline(
  detector: await createFaceDetectorProcessor(args.detectionModel),
  mesh: await createFaceMeshProcessor(
    model: args.meshMode.model,
    iris: args.meshMode.enableIris,
  ),
  landmarkSmoothing: landmarkSmoothing,
);
