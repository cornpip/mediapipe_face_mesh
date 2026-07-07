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
  ///
  /// Null when the frame was served by landmark tracking, in which case the
  /// detector did not run and [meshResult] came from the mesh processor's
  /// internal ROI tracking.
  final FaceDetectionResult? detectionResult;

  /// Detection selected for mesh inference.
  ///
  /// Null on landmark-tracked frames and on frames without a usable face.
  final FaceDetection? selectedDetection;

  /// ROI used for face mesh inference.
  ///
  /// On detector-driven frames this is the ROI derived from
  /// [selectedDetection]; on landmark-tracked frames it is the tracked ROI
  /// reported by the mesh processor. When [meshResult] is null, this ROI may
  /// have been selected but not used.
  final NormalizedRect? selectedRoi;

  /// Pixel-space box for [selectedDetection].
  FaceMeshBox? get selectedBox {
    final FaceDetectionResult? detections = detectionResult;
    if (detections == null) {
      return null;
    }
    return selectedDetection?.toBox(
      imageWidth: detections.imageWidth,
      imageHeight: detections.imageHeight,
    );
  }

  /// Mesh result for the tracked or detected face, or null when no usable
  /// face was found.
  final FaceMeshResult? meshResult;

  /// Whether the detector ran for this frame.
  ///
  /// False on frames served by landmark tracking.
  bool get detectorRan => detectionResult != null;

  /// Whether the detector selected a face.
  ///
  /// False on landmark-tracked frames even though a face is being followed;
  /// check [hasMesh] for face presence in that mode.
  bool get hasFace => selectedDetection != null;

  /// Whether a mesh ROI was available for this frame.
  bool get hasRoi => selectedRoi != null;

  /// Whether face mesh inference produced landmarks.
  bool get hasMesh => meshResult != null;

  @override
  String toString() =>
      'FaceMeshInferenceResult(detectorRan: $detectorRan, detections: '
      '${detectionResult?.detections.length ?? 0}, hasFace: $hasFace, '
      'hasRoi: $hasRoi, hasMesh: $hasMesh)';
}

/// A face mesh followed across frames by the multi-face flow.
class TrackedFaceMesh {
  /// Creates a tracked face entry.
  const TrackedFaceMesh({required this.trackId, required this.mesh});

  /// Identifier that stays stable for as long as this face is tracked.
  ///
  /// A face that is lost and later re-acquired receives a new id. When
  /// landmark tracking is disabled, this is just the index within the frame.
  final int trackId;

  /// Mesh result for this face.
  final FaceMeshResult mesh;

  @override
  String toString() =>
      'TrackedFaceMesh(trackId: $trackId, landmarks: '
      '${mesh.landmarks.length})';
}

/// Result of running face detection and multi-face mesh inference together.
class FaceMeshMultiInferenceResult {
  /// Creates a multi-face inference result.
  const FaceMeshMultiInferenceResult({
    required this.detectionResult,
    required this.faces,
  });

  /// Raw detector output for the input frame.
  ///
  /// Null when every tracked face slot was filled by landmark tracking and
  /// the detector did not run.
  final FaceDetectionResult? detectionResult;

  /// Tracked faces with their mesh results.
  ///
  /// With landmark tracking, entries keep their [TrackedFaceMesh.trackId]
  /// across frames; otherwise they follow detector score order.
  final List<TrackedFaceMesh> faces;

  /// Mesh results for [faces], in the same order.
  List<FaceMeshResult> get meshResults => <FaceMeshResult>[
    for (final TrackedFaceMesh face in faces) face.mesh,
  ];

  /// Whether the detector ran for this frame.
  bool get detectorRan => detectionResult != null;

  /// Whether this frame produced at least one face (detected or tracked).
  bool get hasFaces =>
      faces.isNotEmpty || (detectionResult?.detections.isNotEmpty ?? false);

  /// Whether face mesh inference produced at least one mesh.
  bool get hasMeshes => faces.isNotEmpty;

  @override
  String toString() =>
      'FaceMeshMultiInferenceResult(detectorRan: $detectorRan, detections: '
      '${detectionResult?.detections.length ?? 0}, faces: ${faces.length})';
}

/// Runs face detection and face mesh inference as one flow.
///
/// By default the single-face flow mirrors the official MediaPipe Face Mesh
/// graph: the detector runs only to (re)acquire a face, and while a face is
/// being followed the mesh ROI comes from the previous frame's landmarks via
/// the mesh processor's internal ROI tracking. This keeps the mesh accurate
/// when the detection box is imprecise (for example with a wide-open mouth)
/// and skips the detector entirely on tracked frames.
///
/// The multi-face methods track the same way with one ROI per face: each
/// tracked face keeps a stable [TrackedFaceMesh.trackId], and the detector
/// runs only while fewer than `maxMeshFaces` faces are tracked. Multi-face
/// tracking is managed in Dart, so it works with a
/// [FaceMeshProcessor.createForMultiFace] processor. The two flows share the
/// native mesh state, so calling one resets the other's tracking and the
/// next call of the other flow re-acquires via the detector.
///
/// Single-face landmark tracking requires a [mesh] processor created with
/// `enableRoiTracking: true` (the default). Passing
/// `enableLandmarkTracking: false` to [FaceMeshInferencePipeline.new] makes
/// every frame run detector-driven as before, in both flows.
///
/// On tracked frames the detector does not run, so `detectorRoi` and the
/// detector ROI scale/shift overrides apply only to (re)acquisition frames —
/// a face acquired inside a restricted region keeps being tracked after it
/// leaves that region. Tracking resets automatically when the input type,
/// frame size, rotation, or mirroring changes; call [resetTracking] when
/// switching between sources the pipeline cannot tell apart (for example two
/// cameras with the same resolution and rotation) so tracking does not resume
/// on a stale region of the new feed.
///
/// The pipeline does not own [detector] or [mesh]. Close those processors
/// directly when they are no longer needed.
class FaceMeshInferencePipeline {
  /// Creates a face mesh pipeline.
  ///
  /// Set [enableLandmarkTracking] to false to run the detector on every
  /// frame and always derive the mesh ROI from the detection result.
  FaceMeshInferencePipeline({
    required FaceDetectorProcessor detector,
    required FaceMeshProcessor mesh,
    FaceDetectionSelector? detectionSelector,
    bool enableLandmarkTracking = true,
  }) : _detector = detector,
       _mesh = mesh,
       _detectionSelector = detectionSelector ?? _defaultDetectionSelector,
       _landmarkTrackingEnabled =
           enableLandmarkTracking && mesh.roiTrackingEnabled,
       _multiTrackingEnabled = enableLandmarkTracking;

  /// Minimum IoU between a detection ROI and a tracked face ROI for the two
  /// to be considered the same face — matches the official
  /// `AssociationNormRectCalculator` threshold.
  static const double _trackAssociationIou = 0.5;

  final FaceDetectorProcessor _detector;
  final FaceMeshProcessor _mesh;
  final FaceDetectionSelector _detectionSelector;
  final bool _landmarkTrackingEnabled;

  /// Multi-face tracking runs in Dart with explicit per-face ROIs, so unlike
  /// the single-face flow it does not need the mesh processor's native ROI
  /// tracking.
  final bool _multiTrackingEnabled;

  bool _isTracking = false;
  final List<_FaceTrack> _multiTracks = <_FaceTrack>[];
  int _nextTrackId = 0;
  _FaceMeshInputKind? _lastInputKind;
  int? _lastInputWidth;
  int? _lastInputHeight;
  int _lastRotationDegrees = 0;
  bool _lastMirrorHorizontal = false;

  /// Whether the single-face flow is currently following a face via landmark
  /// tracking instead of running the detector.
  bool get isTracking => _isTracking;

  /// Track ids of the faces the multi-face flow is currently following.
  List<int> get trackedFaceIds => <int>[
    for (final _FaceTrack track in _multiTracks) track.id,
  ];

  /// Forces the next frame to re-run the detector, dropping all tracked
  /// faces.
  ///
  /// Tracking also resets automatically when the input type, frame size,
  /// rotation, or mirroring changes; call this for source switches the
  /// pipeline cannot detect (for example two cameras with identical frames).
  void resetTracking() {
    _resetTrackingState();
  }

  void _resetTrackingState() {
    _isTracking = false;
    _multiTracks.clear();
  }

  void _syncInput({
    required _FaceMeshInputKind inputKind,
    required int width,
    required int height,
    required int rotationDegrees,
    required bool mirrorHorizontal,
  }) {
    if (inputKind != _lastInputKind ||
        width != _lastInputWidth ||
        height != _lastInputHeight ||
        rotationDegrees != _lastRotationDegrees ||
        mirrorHorizontal != _lastMirrorHorizontal) {
      _resetTrackingState();
      _lastInputKind = inputKind;
      _lastInputWidth = width;
      _lastInputHeight = height;
      _lastRotationDegrees = rotationDegrees;
      _lastMirrorHorizontal = mirrorHorizontal;
    }
  }

  /// Runs the tracked-frame fast path, returning null when the frame must be
  /// served by the detector instead.
  FaceMeshInferenceResult? _tryTrackedFrame(
    bool runMesh,
    FaceMeshResult Function() runTrackedMesh,
  ) {
    if (!runMesh || !_isTracking) {
      return null;
    }
    final FaceMeshResult tracked;
    try {
      tracked = runTrackedMesh();
    } catch (_) {
      // Fall back to detector re-acquisition on the next frame instead of
      // retrying the failing tracked call forever.
      _isTracking = false;
      rethrow;
    }
    if (tracked.landmarks.isEmpty) {
      _isTracking = false;
      return null;
    }
    return FaceMeshInferenceResult(
      detectionResult: null,
      selectedDetection: null,
      selectedRoi: tracked.rect,
      meshResult: tracked,
    );
  }

  void _updateTracking(FaceMeshResult? meshResult) {
    _isTracking =
        _landmarkTrackingEnabled &&
        meshResult != null &&
        meshResult.landmarks.isNotEmpty;
  }

  /// Shared multi-face flow: advance tracked faces, then acquire new faces
  /// via the detector while slots are free.
  FaceMeshMultiInferenceResult _processMulti({
    required bool runMesh,
    required int? maxMeshFaces,
    required _FaceMeshInputKind inputKind,
    required int width,
    required int height,
    required int rotationDegrees,
    required bool mirrorHorizontal,
    required FaceDetectionResult Function() runDetector,
    required FaceMeshResult Function(NormalizedRect roi) runMeshWithRoi,
    required List<FaceMeshResult> Function(FaceDetectionResult detectionResult)
    runLegacyMultiMesh,
  }) {
    _validateMaxMeshFaces(maxMeshFaces);
    _syncInput(
      inputKind: inputKind,
      width: width,
      height: height,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
    );
    // The per-face mesh calls overwrite the native tracked ROI, so force the
    // next single-face frame to re-acquire via the detector.
    _isTracking = false;

    if (!runMesh) {
      _multiTracks.clear();
      return FaceMeshMultiInferenceResult(
        detectionResult: runDetector(),
        faces: const <TrackedFaceMesh>[],
      );
    }

    if (!_multiTrackingEnabled) {
      final FaceDetectionResult detectionResult = runDetector();
      final List<FaceMeshResult> meshes = runLegacyMultiMesh(detectionResult);
      return FaceMeshMultiInferenceResult(
        detectionResult: detectionResult,
        faces: <TrackedFaceMesh>[
          for (int i = 0; i < meshes.length; i++)
            TrackedFaceMesh(trackId: i, mesh: meshes[i]),
        ],
      );
    }

    if (maxMeshFaces != null && _multiTracks.length > maxMeshFaces) {
      _multiTracks.removeRange(maxMeshFaces, _multiTracks.length);
    }

    // Advance every tracked face on its landmark-derived ROI; drop the ones
    // that lost their face or whose presence score fell below the tracking
    // confidence, so their slot is re-acquired via the detector (official
    // tracking-confidence semantics).
    final double minTrackingConfidence = _mesh.minTrackingConfidence;
    final List<_FaceTrack> survivors = <_FaceTrack>[];
    try {
      for (final _FaceTrack track in _multiTracks) {
        final FaceMeshResult mesh = runMeshWithRoi(track.roi);
        if (mesh.landmarks.isEmpty || mesh.score < minTrackingConfidence) {
          continue;
        }
        track.mesh = mesh;
        track.roi = mesh.trackingRoi();
        survivors.add(track);
      }
      _multiTracks
        ..clear()
        ..addAll(survivors);

      // Acquire new faces only while slots are free, like the official
      // graph's detector gate.
      FaceDetectionResult? detectionResult;
      if (maxMeshFaces == null || _multiTracks.length < maxMeshFaces) {
        detectionResult = runDetector();
        for (final FaceDetection detection in detectionResult.detections) {
          if (maxMeshFaces != null && _multiTracks.length >= maxMeshFaces) {
            break;
          }
          final NormalizedRect? roi = _roiForDetection(detection);
          if (roi == null || _overlapsTrackedFace(roi)) {
            continue;
          }
          final FaceMeshResult mesh = runMeshWithRoi(roi);
          if (mesh.landmarks.isEmpty) {
            continue;
          }
          _multiTracks.add(
            _FaceTrack(_nextTrackId++, mesh, mesh.trackingRoi()),
          );
        }
      }
      return FaceMeshMultiInferenceResult(
        detectionResult: detectionResult,
        faces: <TrackedFaceMesh>[
          for (final _FaceTrack track in _multiTracks)
            TrackedFaceMesh(trackId: track.id, mesh: track.mesh),
        ],
      );
    } catch (_) {
      // Fall back to detector re-acquisition on the next frame instead of
      // retrying failing tracked calls forever.
      _multiTracks.clear();
      rethrow;
    }
  }

  bool _overlapsTrackedFace(NormalizedRect roi) {
    for (final _FaceTrack track in _multiTracks) {
      if (_rectIou(roi, track.roi) > _trackAssociationIou) {
        return true;
      }
    }
    return false;
  }

  /// IoU of the rects' axis-aligned bounds (rotation ignored), matching the
  /// overlap test the official association calculator uses.
  static double _rectIou(NormalizedRect a, NormalizedRect b) {
    final double left = math.max(
      a.xCenter - a.width / 2,
      b.xCenter - b.width / 2,
    );
    final double top = math.max(
      a.yCenter - a.height / 2,
      b.yCenter - b.height / 2,
    );
    final double right = math.min(
      a.xCenter + a.width / 2,
      b.xCenter + b.width / 2,
    );
    final double bottom = math.min(
      a.yCenter + a.height / 2,
      b.yCenter + b.height / 2,
    );
    final double intersection =
        math.max(0, right - left) * math.max(0, bottom - top);
    final double union = a.width * a.height + b.width * b.height - intersection;
    if (union <= 0) {
      return 0;
    }
    return intersection / union;
  }

  static FaceDetection? _defaultDetectionSelector(FaceDetectionResult result) =>
      result.primaryDetection;

  /// Processes an RGBA/BGRA frame through detector and mesh inference.
  ///
  /// While a face is tracked (see [isTracking]) the detector is skipped and
  /// the mesh runs on its internally tracked ROI; when tracking is lost the
  /// detector re-acquires the face within the same call. [detectorRoi]
  /// restricts the detector input. On detector-driven frames the selected
  /// detection's [FaceDetection.expandedFaceRect], or [FaceDetection.faceRect]
  /// when the expanded ROI is unavailable, is used as the mesh ROI.
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
    _syncInput(
      inputKind: _FaceMeshInputKind.image,
      width: image.width,
      height: image.height,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
    );
    // Mirror of the multi-face reset: drop multi-face tracks so a later
    // multi-face call re-acquires via the detector instead of advancing
    // stale ROIs.
    _multiTracks.clear();
    final FaceMeshInferenceResult? trackedResult = _tryTrackedFrame(
      runMesh,
      () => _mesh.process(
        image,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
      ),
    );
    if (trackedResult != null) {
      return trackedResult;
    }

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
    _updateTracking(meshResult);

    return FaceMeshInferenceResult(
      detectionResult: detectionResult,
      selectedDetection: selectedDetection,
      selectedRoi: selectedRoi,
      meshResult: meshResult,
    );
  }

  /// Processes an NV21 frame through detector and mesh inference.
  ///
  /// While a face is tracked (see [isTracking]) the detector is skipped and
  /// the mesh runs on its internally tracked ROI; when tracking is lost the
  /// detector re-acquires the face within the same call. [detectorRoi]
  /// restricts the detector input. On detector-driven frames the selected
  /// detection's [FaceDetection.expandedFaceRect], or [FaceDetection.faceRect]
  /// when the expanded ROI is unavailable, is used as the mesh ROI.
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
    _syncInput(
      inputKind: _FaceMeshInputKind.nv21,
      width: image.width,
      height: image.height,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
    );
    // Mirror of the multi-face reset: drop multi-face tracks so a later
    // multi-face call re-acquires via the detector instead of advancing
    // stale ROIs.
    _multiTracks.clear();
    final FaceMeshInferenceResult? trackedResult = _tryTrackedFrame(
      runMesh,
      () => _mesh.processNv21(
        image,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
      ),
    );
    if (trackedResult != null) {
      return trackedResult;
    }

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
    _updateTracking(meshResult);

    return FaceMeshInferenceResult(
      detectionResult: detectionResult,
      selectedDetection: selectedDetection,
      selectedRoi: selectedRoi,
      meshResult: meshResult,
    );
  }

  /// Processes an RGBA/BGRA frame through multi-face detection and mesh
  /// inference.
  ///
  /// With landmark tracking (the default), each tracked face runs mesh
  /// inference on an ROI derived from its previous frame's landmarks, and the
  /// detector runs only while fewer than [maxMeshFaces] faces are tracked —
  /// newly detected faces that do not overlap a tracked face are added with a
  /// new [TrackedFaceMesh.trackId]. With tracking disabled, every frame runs
  /// the detector and one mesh inference per detection, in score order.
  ///
  /// A tracked face is dropped — freeing its slot for detector
  /// re-acquisition — when its mesh presence score falls below the mesh
  /// processor's [FaceMeshProcessor.minTrackingConfidence].
  ///
  /// Set [runMesh] to false to run detector-only (this also drops all tracked
  /// faces). [maxMeshFaces] bounds the number of simultaneously tracked
  /// faces; configure the detector result count separately with
  /// [FaceDetectorProcessor.create]'s `maxResults`.
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
    return _processMulti(
      runMesh: runMesh,
      maxMeshFaces: maxMeshFaces,
      inputKind: _FaceMeshInputKind.image,
      width: image.width,
      height: image.height,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      runDetector: () => _detector.process(
        image,
        roi: detectorRoi,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
        roiScaleX: detectorRoiScaleX,
        roiScaleY: detectorRoiScaleY,
        roiShiftX: detectorRoiShiftX,
        roiShiftY: detectorRoiShiftY,
      ),
      runMeshWithRoi: (NormalizedRect roi) => _mesh.process(
        image,
        roi: roi,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
      ),
      runLegacyMultiMesh: (FaceDetectionResult detectionResult) =>
          _mesh.processMultiFace(
            image,
            detections: detectionResult.detections,
            maxMeshFaces: maxMeshFaces,
            rotationDegrees: rotationDegrees,
            mirrorHorizontal: mirrorHorizontal,
          ),
    );
  }

  /// Processes an NV21 frame through multi-face detection and mesh inference.
  ///
  /// This is the NV21 counterpart of [processMultiFace]; see that method for
  /// the landmark-tracking behavior.
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
    return _processMulti(
      runMesh: runMesh,
      maxMeshFaces: maxMeshFaces,
      inputKind: _FaceMeshInputKind.nv21,
      width: image.width,
      height: image.height,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      runDetector: () => _detector.processNv21(
        image,
        roi: detectorRoi,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
        roiScaleX: detectorRoiScaleX,
        roiScaleY: detectorRoiScaleY,
        roiShiftX: detectorRoiShiftX,
        roiShiftY: detectorRoiShiftY,
      ),
      runMeshWithRoi: (NormalizedRect roi) => _mesh.processNv21(
        image,
        roi: roi,
        rotationDegrees: rotationDegrees,
        mirrorHorizontal: mirrorHorizontal,
      ),
      runLegacyMultiMesh: (FaceDetectionResult detectionResult) =>
          _mesh.processNv21MultiFace(
            image,
            detections: detectionResult.detections,
            maxMeshFaces: maxMeshFaces,
            rotationDegrees: rotationDegrees,
            mirrorHorizontal: mirrorHorizontal,
          ),
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

enum _FaceMeshInputKind { image, nv21 }

/// State for one face followed by the multi-face tracking flow.
class _FaceTrack {
  _FaceTrack(this.id, this.mesh, this.roi);

  final int id;
  FaceMeshResult mesh;

  /// ROI to run the next frame's mesh inference on, derived from [mesh]'s
  /// landmarks.
  NormalizedRect roi;
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
