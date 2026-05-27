part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Selects the detection that should be used for face mesh inference.
typedef FaceDetectionSelector =
    FaceDetection? Function(FaceDetectionResult result);

/// Resolves a detector ROI for a high-level inference frame.
typedef FaceMeshInferenceDetectorRoiResolver<T> =
    NormalizedRect? Function(T frame);

/// Decides whether face mesh inference should run for a frame.
typedef FaceMeshRunResolver<T> = bool Function(T frame);

/// Result of running face detection and face mesh inference as one operation.
class FaceMeshInferenceResult {
  /// Creates a high-level inference result.
  const FaceMeshInferenceResult({
    required this.detectionResult,
    required this.selectedDetection,
    required this.selectedRoi,
    required this.meshResult,
  });

  /// Raw detector output for the input frame.
  final FaceDetectionResult detectionResult;

  /// Detection selected for mesh inference.
  final FaceDetection? selectedDetection;

  /// ROI selected for face mesh inference.
  ///
  /// When [meshResult] is null, this ROI may have been selected but not used.
  final NormalizedRect? selectedRoi;

  /// Pixel-space box for [selectedDetection].
  FaceMeshBox? get selectedBox => selectedDetection?.toBox(
    imageWidth: detectionResult.imageWidth,
    imageHeight: detectionResult.imageHeight,
  );

  /// Mesh result for [selectedDetection], or null when no usable face was found.
  final FaceMeshResult? meshResult;

  /// Whether the detector selected a face.
  bool get hasFace => selectedDetection != null;

  /// Whether the selected detection produced a mesh ROI.
  bool get hasRoi => selectedRoi != null;

  /// Whether face mesh inference produced landmarks.
  bool get hasMesh => meshResult != null;

  @override
  String toString() =>
      'FaceMeshInferenceResult(detections: '
      '${detectionResult.detections.length}, hasFace: $hasFace, '
      'hasRoi: $hasRoi, hasMesh: $hasMesh)';
}

/// Result of running face detection and multi-face mesh inference together.
class FaceMeshMultiInferenceResult {
  /// Creates a multi-face inference result.
  const FaceMeshMultiInferenceResult({
    required this.detectionResult,
    required this.meshResults,
  });

  /// Raw detector output for the input frame.
  final FaceDetectionResult detectionResult;

  /// Mesh results produced from detector ROIs, in detector score order.
  final List<FaceMeshResult> meshResults;

  /// Whether the detector returned at least one face.
  bool get hasFaces => detectionResult.detections.isNotEmpty;

  /// Whether any detected face has a ROI that can be used for mesh inference.
  bool get hasMeshRoi =>
      detectionResult.detections.any((detection) => _roiFor(detection) != null);

  /// Whether face mesh inference produced at least one mesh.
  bool get hasMeshes => meshResults.isNotEmpty;

  @override
  String toString() =>
      'FaceMeshMultiInferenceResult(detections: '
      '${detectionResult.detections.length}, meshes: ${meshResults.length})';

  static NormalizedRect? _roiFor(FaceDetection detection) =>
      detection.expandedFaceRect ?? detection.faceRect;
}

/// Runs face detection and face mesh inference in one detector-driven flow.
///
/// The pipeline does not own [detector] or [mesh]. Close those processors
/// directly when they are no longer needed.
class FaceMeshInferencePipeline {
  /// Creates a detector-driven face mesh pipeline.
  FaceMeshInferencePipeline({
    required FaceDetectorProcessor detector,
    required FaceMeshProcessor mesh,
    FaceDetectionSelector? detectionSelector,
  }) : _detector = detector,
       _mesh = mesh,
       _detectionSelector = detectionSelector ?? _defaultDetectionSelector;

  final FaceDetectorProcessor _detector;
  final FaceMeshProcessor _mesh;
  final FaceDetectionSelector _detectionSelector;

  static FaceDetection? _defaultDetectionSelector(FaceDetectionResult result) =>
      result.primaryDetection;

  /// Processes an RGBA/BGRA frame through detector and mesh inference.
  ///
  /// [detectorRoi] restricts the detector input. The selected detection's
  /// [FaceDetection.expandedFaceRect], or [FaceDetection.faceRect] when the
  /// expanded ROI is unavailable, is then used as the mesh ROI.
  /// Set [runMesh] to false to return detector output without running mesh
  /// inference.
  FaceMeshInferenceResult process(
    FaceMeshImage image, {
    NormalizedRect? detectorRoi,
    bool runMesh = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) {
    final FaceDetectionResult detectionResult = _detector.process(
      image,
      roi: detectorRoi,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      roiScaleX: detectorRoiScaleX,
      roiScaleY: detectorRoiScaleY,
      roiShiftX: detectorRoiShiftX,
      roiShiftY: detectorRoiShiftY,
    );
    final FaceDetection? selectedDetection = _detectionSelector(
      detectionResult,
    );
    final NormalizedRect? selectedRoi = _roiForDetection(selectedDetection);
    final FaceMeshResult? meshResult = !runMesh || selectedRoi == null
        ? null
        : _mesh.process(
            image,
            roi: selectedRoi,
            rotationDegrees: rotationDegrees,
            mirrorHorizontal: mirrorHorizontal,
          );

    return FaceMeshInferenceResult(
      detectionResult: detectionResult,
      selectedDetection: selectedDetection,
      selectedRoi: selectedRoi,
      meshResult: meshResult,
    );
  }

  /// Processes an NV21 frame through detector and mesh inference.
  ///
  /// [detectorRoi] restricts the detector input. The selected detection's
  /// [FaceDetection.expandedFaceRect], or [FaceDetection.faceRect] when the
  /// expanded ROI is unavailable, is then used as the mesh ROI.
  /// Set [runMesh] to false to return detector output without running mesh
  /// inference.
  FaceMeshInferenceResult processNv21(
    FaceMeshNv21Image image, {
    NormalizedRect? detectorRoi,
    bool runMesh = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) {
    final FaceDetectionResult detectionResult = _detector.processNv21(
      image,
      roi: detectorRoi,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      roiScaleX: detectorRoiScaleX,
      roiScaleY: detectorRoiScaleY,
      roiShiftX: detectorRoiShiftX,
      roiShiftY: detectorRoiShiftY,
    );
    final FaceDetection? selectedDetection = _detectionSelector(
      detectionResult,
    );
    final NormalizedRect? selectedRoi = _roiForDetection(selectedDetection);
    final FaceMeshResult? meshResult = !runMesh || selectedRoi == null
        ? null
        : _mesh.processNv21(
            image,
            roi: selectedRoi,
            rotationDegrees: rotationDegrees,
            mirrorHorizontal: mirrorHorizontal,
          );

    return FaceMeshInferenceResult(
      detectionResult: detectionResult,
      selectedDetection: selectedDetection,
      selectedRoi: selectedRoi,
      meshResult: meshResult,
    );
  }

  /// Processes an RGBA/BGRA frame through detector inference, then runs mesh
  /// inference once for each detector result with a usable ROI.
  ///
  /// The returned mesh list preserves detector score order. Set [runMesh] to
  /// false to run detector-only and return an empty mesh list.
  ///
  /// [maxMeshFaces] limits how many mesh inferences are run. Configure the
  /// detector result count separately with [FaceDetectorProcessor.create]'s
  /// `maxResults`.
  FaceMeshMultiInferenceResult processMultiFace(
    FaceMeshImage image, {
    NormalizedRect? detectorRoi,
    bool runMesh = true,
    int? maxMeshFaces,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) {
    _validateMaxMeshFaces(maxMeshFaces);
    final FaceDetectionResult detectionResult = _detector.process(
      image,
      roi: detectorRoi,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      roiScaleX: detectorRoiScaleX,
      roiScaleY: detectorRoiScaleY,
      roiShiftX: detectorRoiShiftX,
      roiShiftY: detectorRoiShiftY,
    );
    final List<FaceMeshResult> meshResults = runMesh
        ? _mesh.processMultiFace(
            image,
            detections: detectionResult.detections,
            maxMeshFaces: maxMeshFaces,
            rotationDegrees: rotationDegrees,
            mirrorHorizontal: mirrorHorizontal,
          )
        : <FaceMeshResult>[];
    return FaceMeshMultiInferenceResult(
      detectionResult: detectionResult,
      meshResults: meshResults,
    );
  }

  /// Processes an NV21 frame through detector inference, then runs mesh
  /// inference once for each detector result with a usable ROI.
  ///
  /// The returned mesh list preserves detector score order. Set [runMesh] to
  /// false to run detector-only and return an empty mesh list.
  ///
  /// [maxMeshFaces] limits how many mesh inferences are run. Configure the
  /// detector result count separately with [FaceDetectorProcessor.create]'s
  /// `maxResults`.
  FaceMeshMultiInferenceResult processNv21MultiFace(
    FaceMeshNv21Image image, {
    NormalizedRect? detectorRoi,
    bool runMesh = true,
    int? maxMeshFaces,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) {
    _validateMaxMeshFaces(maxMeshFaces);
    final FaceDetectionResult detectionResult = _detector.processNv21(
      image,
      roi: detectorRoi,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      roiScaleX: detectorRoiScaleX,
      roiScaleY: detectorRoiScaleY,
      roiShiftX: detectorRoiShiftX,
      roiShiftY: detectorRoiShiftY,
    );
    final List<FaceMeshResult> meshResults = runMesh
        ? _mesh.processNv21MultiFace(
            image,
            detections: detectionResult.detections,
            maxMeshFaces: maxMeshFaces,
            rotationDegrees: rotationDegrees,
            mirrorHorizontal: mirrorHorizontal,
          )
        : <FaceMeshResult>[];
    return FaceMeshMultiInferenceResult(
      detectionResult: detectionResult,
      meshResults: meshResults,
    );
  }

  NormalizedRect? _roiForDetection(FaceDetection? detection) =>
      detection?.expandedFaceRect ?? detection?.faceRect;

  void _validateMaxMeshFaces(int? maxMeshFaces) {
    if (maxMeshFaces != null && maxMeshFaces < 0) {
      throw ArgumentError('maxMeshFaces must be null or >= 0.');
    }
  }
}

/// Helper that turns a stream of frames into high-level inference results.
class FaceMeshInferenceStreamProcessor {
  /// Creates a stream processor bound to [pipeline].
  FaceMeshInferenceStreamProcessor(this._pipeline);

  final FaceMeshInferencePipeline _pipeline;

  /// Processes a stream of RGBA/BGRA frames sequentially.
  ///
  /// [detectorRoi] restricts every detector invocation. Use
  /// [detectorRoiResolver] when the detector ROI changes per frame.
  /// Set [runMesh] to false to run detector-only for the whole stream, or
  /// return false from [runMeshResolver] to run detector-only for a frame.
  Stream<FaceMeshInferenceResult> process(
    Stream<FaceMeshImage> frames, {
    NormalizedRect? detectorRoi,
    FaceMeshInferenceDetectorRoiResolver<FaceMeshImage>? detectorRoiResolver,
    bool runMesh = true,
    FaceMeshRunResolver<FaceMeshImage>? runMeshResolver,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) async* {
    _validateResolvers<FaceMeshImage>(detectorRoi, detectorRoiResolver);
    _validateRunMeshOptions<FaceMeshImage>(runMesh, runMeshResolver);
    await for (final FaceMeshImage frame in frames) {
      final NormalizedRect? dynamicDetectorRoi = detectorRoiResolver?.call(
        frame,
      );
      final bool shouldRunMesh = runMeshResolver?.call(frame) ?? runMesh;
      yield _pipeline.process(
        frame,
        detectorRoi: dynamicDetectorRoi ?? detectorRoi,
        runMesh: shouldRunMesh,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
        detectorRoiScaleX: detectorRoiScaleX,
        detectorRoiScaleY: detectorRoiScaleY,
        detectorRoiShiftX: detectorRoiShiftX,
        detectorRoiShiftY: detectorRoiShiftY,
      );
    }
  }

  /// Processes a stream of NV21 frames sequentially.
  ///
  /// [detectorRoi] restricts every detector invocation. Use
  /// [detectorRoiResolver] when the detector ROI changes per frame.
  /// Set [runMesh] to false to run detector-only for the whole stream, or
  /// return false from [runMeshResolver] to run detector-only for a frame.
  Stream<FaceMeshInferenceResult> processNv21(
    Stream<FaceMeshNv21Image> frames, {
    NormalizedRect? detectorRoi,
    FaceMeshInferenceDetectorRoiResolver<FaceMeshNv21Image>?
    detectorRoiResolver,
    bool runMesh = true,
    FaceMeshRunResolver<FaceMeshNv21Image>? runMeshResolver,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) async* {
    _validateResolvers<FaceMeshNv21Image>(detectorRoi, detectorRoiResolver);
    _validateRunMeshOptions<FaceMeshNv21Image>(runMesh, runMeshResolver);
    await for (final FaceMeshNv21Image frame in frames) {
      final NormalizedRect? dynamicDetectorRoi = detectorRoiResolver?.call(
        frame,
      );
      final bool shouldRunMesh = runMeshResolver?.call(frame) ?? runMesh;
      yield _pipeline.processNv21(
        frame,
        detectorRoi: dynamicDetectorRoi ?? detectorRoi,
        runMesh: shouldRunMesh,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
        detectorRoiScaleX: detectorRoiScaleX,
        detectorRoiScaleY: detectorRoiScaleY,
        detectorRoiShiftX: detectorRoiShiftX,
        detectorRoiShiftY: detectorRoiShiftY,
      );
    }
  }

  /// Processes a stream of RGBA/BGRA frames into multi-face mesh results.
  ///
  /// Each frame runs detector inference once, then mesh inference for each
  /// detector result with a usable ROI. [maxMeshFaces] limits mesh invocations
  /// per frame; detector result count is controlled by the detector processor's
  /// `maxResults` option.
  Stream<FaceMeshMultiInferenceResult> processMultiFace(
    Stream<FaceMeshImage> frames, {
    NormalizedRect? detectorRoi,
    FaceMeshInferenceDetectorRoiResolver<FaceMeshImage>? detectorRoiResolver,
    bool runMesh = true,
    FaceMeshRunResolver<FaceMeshImage>? runMeshResolver,
    int? maxMeshFaces,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) async* {
    _validateResolvers<FaceMeshImage>(detectorRoi, detectorRoiResolver);
    _validateRunMeshOptions<FaceMeshImage>(runMesh, runMeshResolver);
    await for (final FaceMeshImage frame in frames) {
      final NormalizedRect? dynamicDetectorRoi = detectorRoiResolver?.call(
        frame,
      );
      final bool shouldRunMesh = runMeshResolver?.call(frame) ?? runMesh;
      yield _pipeline.processMultiFace(
        frame,
        detectorRoi: dynamicDetectorRoi ?? detectorRoi,
        runMesh: shouldRunMesh,
        maxMeshFaces: maxMeshFaces,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
        detectorRoiScaleX: detectorRoiScaleX,
        detectorRoiScaleY: detectorRoiScaleY,
        detectorRoiShiftX: detectorRoiShiftX,
        detectorRoiShiftY: detectorRoiShiftY,
      );
    }
  }

  /// Processes a stream of NV21 frames into multi-face mesh results.
  ///
  /// Each frame runs detector inference once, then mesh inference for each
  /// detector result with a usable ROI. [maxMeshFaces] limits mesh invocations
  /// per frame; detector result count is controlled by the detector processor's
  /// `maxResults` option.
  Stream<FaceMeshMultiInferenceResult> processNv21MultiFace(
    Stream<FaceMeshNv21Image> frames, {
    NormalizedRect? detectorRoi,
    FaceMeshInferenceDetectorRoiResolver<FaceMeshNv21Image>?
    detectorRoiResolver,
    bool runMesh = true,
    FaceMeshRunResolver<FaceMeshNv21Image>? runMeshResolver,
    int? maxMeshFaces,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
  }) async* {
    _validateResolvers<FaceMeshNv21Image>(detectorRoi, detectorRoiResolver);
    _validateRunMeshOptions<FaceMeshNv21Image>(runMesh, runMeshResolver);
    await for (final FaceMeshNv21Image frame in frames) {
      final NormalizedRect? dynamicDetectorRoi = detectorRoiResolver?.call(
        frame,
      );
      final bool shouldRunMesh = runMeshResolver?.call(frame) ?? runMesh;
      yield _pipeline.processNv21MultiFace(
        frame,
        detectorRoi: dynamicDetectorRoi ?? detectorRoi,
        runMesh: shouldRunMesh,
        maxMeshFaces: maxMeshFaces,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
        detectorRoiScaleX: detectorRoiScaleX,
        detectorRoiScaleY: detectorRoiScaleY,
        detectorRoiShiftX: detectorRoiShiftX,
        detectorRoiShiftY: detectorRoiShiftY,
      );
    }
  }

  void _validateResolvers<T>(
    NormalizedRect? detectorRoi,
    FaceMeshInferenceDetectorRoiResolver<T>? detectorRoiResolver,
  ) {
    if (detectorRoi != null && detectorRoiResolver != null) {
      throw ArgumentError(
        'Provide only one of detectorRoi or detectorRoiResolver.',
      );
    }
  }

  void _validateRunMeshOptions<T>(
    bool runMesh,
    FaceMeshRunResolver<T>? runMeshResolver,
  ) {
    if (!runMesh && runMeshResolver != null) {
      throw ArgumentError('Set runMesh to true when using runMeshResolver.');
    }
  }
}
