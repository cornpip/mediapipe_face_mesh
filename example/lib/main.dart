import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/face_detection_painter.dart';
import 'package:mediapipe_face_mesh/face_mesh_painter.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'sources/camera_frame_source.dart';
import 'sources/frame_source.dart';
import 'sources/uvc_frame_source.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Windows frames come from a USB (UVC) camera; the camera plugin has no
  // image stream there and orientation control only exists on mobile
  // embedders.
  if (Platform.isWindows) {
    runApp(MyApp(frameSource: UvcFrameSource()));
    return;
  }
  // The demo UI (preview layout and overlay mapping) assumes portrait.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final List<CameraDescription> cameras = await availableCameras();
  runApp(MyApp(frameSource: CameraFrameSource(cameras)));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key, required this.frameSource});

  final DemoFrameSource frameSource;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MediaPipe Face Mesh',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: MediaPipeFacePage(frameSource: frameSource),
    );
  }
}

class MediaPipeFacePage extends StatefulWidget {
  const MediaPipeFacePage({super.key, required this.frameSource});

  final DemoFrameSource frameSource;

  @override
  State<MediaPipeFacePage> createState() => _MediaPipeFacePageState();
}

/// Face mesh model selection: base mesh, base + iris two-pass, or one of the
/// unified 478-landmark models (attention, FaceMesh-V2). A single choice
/// avoids ambiguous combinations.
enum _MeshMode {
  base('Mesh (468)'),
  iris('Mesh (468) + Iris (10)'),
  attention('Attention Mesh (478)'),
  faceMeshV2('FaceMesh-V2 (478, upstream)');

  const _MeshMode(this.label);

  final String label;

  FaceMeshModel get model => switch (this) {
    _MeshMode.base || _MeshMode.iris => FaceMeshModel.v1,
    _MeshMode.attention => FaceMeshModel.attention,
    _MeshMode.faceMeshV2 => FaceMeshModel.v2,
  };

  bool get enableIris => this == _MeshMode.iris;
}

/// Model choices the worker isolate builds its pipeline from. A record of
/// enums, so it is sendable as is.
typedef _IsolatePipelineArgs = ({
  FaceDetectionModel detectionModel,
  _MeshMode meshMode,
});

class _MediaPipeFacePageState extends State<MediaPipeFacePage>
    with WidgetsBindingObserver {
  DemoFrameSource get _frameSource => widget.frameSource;

  /// setState when mounted, plain assignment otherwise. Results and camera
  /// callbacks can land after the page is gone.
  void _update(VoidCallback fn) {
    if (mounted) {
      setState(fn);
    } else {
      fn();
    }
  }

  String? _errorMessage;
  bool _isInitializing = true;
  bool _isCameraActive = false;
  bool _isCameraBusy = false;
  bool _isChangingCamera = false;
  bool _isDetectionActive = false;
  bool _isMeshActive = false;

  /// Run the pipeline in a worker isolate (FaceMeshIsolatePipeline) instead
  /// of on the UI isolate. Rebuilt whenever a model changes.
  bool _runInIsolate = false;
  FaceMeshIsolatePipeline? _isolatePipeline;
  int _isolateGeneration = 0;
  static const Duration _inferenceFpsUpdateInterval = Duration(
    milliseconds: 200,
  );
  double _inferenceFps = 0;
  DateTime? _lastInferenceTime;
  DateTime? _lastInferenceFpsUpdateTime;

  /// Last single-face pipeline result. Null in multi-face mode.
  FaceMeshInferenceResult? _inference;

  /// Last multi-face pipeline result. Null in single-face mode.
  FaceMeshMultiInferenceResult? _multiInference;

  /// Per-track overlay labels for the multi-face ROI boxes. Computed when
  /// the result arrives, since each runs the blendshapes post-processor.
  Map<int, String> _multiFaceLabels = const <int, String>{};

  String? _movementLabel;

  /// Head pose and distances of the current mesh result, built once per
  /// result since estimateGeometry runs a native solve.
  String? _geometryText;
  FaceBlendshapesProcessor? _blendshapesProcessor;
  late FaceDetectorProcessor _faceDetectorProcessor;
  late FaceMeshProcessor _faceMeshProcessor;
  late FaceMeshInferencePipeline _faceMeshInferencePipeline;

  /// Rotation of the last frame handed to inference. Null until the first
  /// frame after _resetInference, which marks a new input source.
  int? _lastInferenceRotation;
  FaceDetectionModel _detectionModel = FaceDetectionModel.shortRange;
  _MeshMode _meshMode = _MeshMode.faceMeshV2;
  bool _isMultiFaceActive = false;

  /// Display-only rotation (0/90/180/270) of the composited preview and
  /// overlays; inference is unaffected, so the mesh stays on the face.
  int _userRotationDegrees = 0;

  /// Display-only horizontal flip of the composited preview and overlays
  /// (selfie-view toggle); inference is unaffected.
  bool _userMirror = false;

  /// Display-only vertical flip of the composited preview and overlays;
  /// inference is unaffected.
  bool _userFlipVertical = false;

  /// Input-side rotation (0/90/180/270) added to the source's rotation
  /// compensation and passed to the pipeline as `rotationDegrees`. Results
  /// come back in the rotated coordinate space and are drawn as-is, so the
  /// mesh visibly rotates. On a correctly oriented source a non-zero value
  /// also makes the model see a sideways face, so detection dropping out is
  /// expected (the control exists to fix wrongly oriented feeds).
  int _inputRotationDegrees = 0;

  /// Input-side mirror passed to the pipeline as `mirrorHorizontal`:
  /// landmark x comes back flipped (selfie coordinate system) and is drawn
  /// as-is, so the mesh visibly mirrors.
  bool _inputMirror = false;

  /// OneEuro landmark smoothing (official FaceLandmarker stream-mode
  /// behavior): removes per-point jitter on a still face while fast head
  /// motion passes through with almost no lag. The demo always enables it.
  static const LandmarkSmoothingOptions _landmarkSmoothing =
      LandmarkSmoothingOptions();
  static const int _maxMeshFaces = 4;
  final ScrollController _controlsScrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _frameSource.onFrame = _handleSourceFrame;
    _frameSource.addListener(_onFrameSourceChanged);
    _initialize();
  }

  void _onFrameSourceChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _initialize() async {
    try {
      _faceDetectorProcessor = await _createFaceDetectorProcessor(
        _detectionModel,
      );

      final faceMeshProcessor = await _createFaceMeshProcessor(
        model: _meshMode.model,
        iris: _meshMode.enableIris,
      );
      // Create the blendshapes processor once (it loads the model), then run it
      // on each mesh result below (the mesh must include iris landmarks).
      _blendshapesProcessor = await FaceBlendshapesProcessor.create(
        delegate: _preferredDelegate,
      );
      final inferencePipeline = FaceMeshInferencePipeline(
        detector: _faceDetectorProcessor,
        mesh: faceMeshProcessor,
        landmarkSmoothing: _landmarkSmoothing,
      );
      _update(() {
        _faceMeshProcessor = faceMeshProcessor;
        _faceMeshInferencePipeline = inferencePipeline;
      });
    } catch (error) {
      _errorMessage = '$error';
    } finally {
      _update(() => _isInitializing = false);
    }
  }

  /// XNNPACK with the default CPU fallback: same speed as cpu on Android,
  /// 4~5x faster on Windows.
  static const FaceMeshDelegate _preferredDelegate = FaceMeshDelegate.xnnpack;

  /// Static so the worker isolate builds its processors with the same
  /// options (see _createPipelineInWorker).
  static Future<FaceDetectorProcessor> _createFaceDetectorProcessor(
    FaceDetectionModel model,
  ) {
    final isFullRange = model != FaceDetectionModel.shortRange;
    return FaceDetectorProcessor.create(
      model: model,
      delegate: _preferredDelegate,
      // Let the detector return several candidates; the single-face flow
      // still picks the best one, and the multi-face flow needs them all.
      maxResults: _maxMeshFaces,
      // Detector ROI defaults are scaleX/scaleY = 1.5 and shiftX/shiftY = 0.0.
      // This demo keeps the default X values and only nudges Y; with landmark
      // tracking these apply to (re)acquisition frames only. Tune per
      // model/camera if the acquisition box is too loose or tight.
      roiScaleY: isFullRange ? 1.6 : 1.7,
      roiShiftY: isFullRange ? -0.1 : -0.2,
    );
  }

  static Future<FaceMeshProcessor> _createFaceMeshProcessor({
    required FaceMeshModel model,
    required bool iris,
  }) async {
    final FaceMeshProcessor processor = await FaceMeshProcessor.create(
      model: model,
      enableIris: iris,
      delegate: _preferredDelegate,
    );
    debugPrint(
      'FaceMeshProcessor created: model=$model iris=$iris '
      'delegate=${processor.activeDelegate}',
    );
    return processor;
  }

  /// Runs inside the worker isolate. Static, so it can be handed to
  /// FaceMeshIsolatePipeline.spawn without capturing this State.
  static Future<FaceMeshInferencePipeline> _createPipelineInWorker(
    _IsolatePipelineArgs args,
  ) async => FaceMeshInferencePipeline(
    detector: await _createFaceDetectorProcessor(args.detectionModel),
    mesh: await _createFaceMeshProcessor(
      model: args.meshMode.model,
      iris: args.meshMode.enableIris,
    ),
    landmarkSmoothing: _landmarkSmoothing,
  );

  Future<void> _changeDetectionModel(FaceDetectionModel value) async {
    if (value == _detectionModel) {
      return;
    }
    final previousSelection = _detectionModel;
    _update(() {
      _detectionModel = value;
      _errorMessage = null;
    });

    try {
      final newFaceDetectorProcessor = await _createFaceDetectorProcessor(
        _detectionModel,
      );
      _resetInference();
      _clearDetections();
      final oldProcessor = _faceDetectorProcessor;
      _faceDetectorProcessor = newFaceDetectorProcessor;
      _faceMeshInferencePipeline = FaceMeshInferencePipeline(
        detector: newFaceDetectorProcessor,
        mesh: _faceMeshProcessor,
        landmarkSmoothing: _landmarkSmoothing,
      );
      oldProcessor.close();
      unawaited(_restartIsolatePipeline());
    } catch (error) {
      _update(() {
        _detectionModel = previousSelection;
        _errorMessage = '$error';
      });
    }
  }

  _IsolatePipelineArgs _currentIsolateArgs() =>
      (detectionModel: _detectionModel, meshMode: _meshMode);

  /// Closes the current worker, if any, and spawns one for the current
  /// models. No-op unless isolate mode is on.
  Future<void> _restartIsolatePipeline() async {
    final FaceMeshIsolatePipeline? old = _isolatePipeline;
    _isolatePipeline = null;
    await old?.close();
    if (!_runInIsolate) {
      return;
    }
    final int generation = ++_isolateGeneration;
    final _IsolatePipelineArgs args = _currentIsolateArgs();
    try {
      final FaceMeshIsolatePipeline pipeline =
          await FaceMeshIsolatePipeline.spawn(_createPipelineInWorker, args);
      if (generation != _isolateGeneration || !_runInIsolate || !mounted) {
        await pipeline.close();
        return;
      }
      _isolatePipeline = pipeline;
    } catch (error, stackTrace) {
      debugPrint('Isolate spawn error: $error\n$stackTrace');
      if (mounted) {
        setState(() => _errorMessage = 'Isolate spawn error: $error');
      }
    }
  }

  Future<void> _toggleRunInIsolate() async {
    _resetInference();
    _clearDetections();
    setState(() => _runInIsolate = !_runInIsolate);
    await _restartIsolatePipeline();
  }

  Future<bool> _startFrameSource() async {
    _clearInferenceFps();
    _resetInference();
    _clearDetections();
    _frameSource.lastError = null;
    final bool started = await _frameSource.start();
    if (!started) {
      _errorMessage = _frameSource.lastError ?? 'Failed to start the camera.';
    }
    if (mounted) {
      setState(() {});
    }
    return started;
  }

  void _updateInferenceFps(DateTime timestamp) {
    final prev = _lastInferenceTime;
    _lastInferenceTime = timestamp;
    if (prev == null) {
      return;
    }
    final elapsed = timestamp.difference(prev).inMicroseconds;
    if (elapsed <= 0) {
      return;
    }
    final fps = 1000000.0 / elapsed;
    final lastUpdate = _lastInferenceFpsUpdateTime;
    if (lastUpdate != null &&
        timestamp.difference(lastUpdate) < _inferenceFpsUpdateInterval) {
      return;
    }
    _lastInferenceFpsUpdateTime = timestamp;
    _update(() => _inferenceFps = fps);
  }

  void _clearInferenceFps() {
    _lastInferenceTime = null;
    _lastInferenceFpsUpdateTime = null;
    _inferenceFps = 0;
  }

  void _clearDetections() {
    _inference = null;
    _multiInference = null;
    _multiFaceLabels = const <int, String>{};
  }

  void _clearMesh() {
    _movementLabel = null;
    _geometryText = null;
  }

  /// Marks the next frame as the first from a new input source (camera
  /// switch, restart, mode change). Inference then drops its tracked face
  /// before that frame.
  void _resetInference() {
    _lastInferenceRotation = null;
  }

  /// True on the first frame after _resetInference. The caller resets
  /// pipeline tracking on that frame.
  bool _noteInputRotation(int rotationDegrees) {
    final bool firstFrame = _lastInferenceRotation == null;
    _lastInferenceRotation = rotationDegrees;
    return firstFrame;
  }

  void _handleInferenceResult(FaceMeshInferenceResult result) {
    if (_lastInferenceRotation == null || !_isDetectionStageActive()) {
      return;
    }
    _updateInferenceFps(DateTime.now());

    _update(() {
      _inference = result;
      _multiInference = null;
      _multiFaceLabels = const <int, String>{};
    });
    _applyMeshStage(result.meshResult);
  }

  void _handleMultiInferenceResult(FaceMeshMultiInferenceResult result) {
    if (_lastInferenceRotation == null || !_isDetectionStageActive()) {
      return;
    }
    _updateInferenceFps(DateTime.now());

    // The movement label runs the blendshapes post-processor per face, same
    // as the single-face movement chip.
    final Map<int, String> labels = <int, String>{
      for (final TrackedFaceMesh face in result.faces)
        face.trackId: switch (_resolveMovementLabel(face.mesh)) {
          null => '#${face.trackId}',
          final String movement => '#${face.trackId} $movement',
        },
    };

    _update(() {
      _inference = null;
      _multiInference = result;
      _multiFaceLabels = labels;
      _clearMesh(); // single-face chips stay off in multi mode
    });
  }

  void _handleInferenceError(Object error) {
    debugPrint('Inference error: $error');
    _update(() => _errorMessage ??= '$error');
  }

  void _applyMeshStage(FaceMeshResult? result) {
    final FaceMeshResult? meshResult = _isMeshActive ? result : null;
    final String? movementLabel = _resolveMovementLabel(meshResult);
    final String? geometryText =
        meshResult != null && meshResult.landmarks.length >= 468
        ? _geometryTextOf(meshResult)
        : null;
    _update(() {
      _movementLabel = movementLabel;
      _geometryText = geometryText;
    });
  }

  /// Runs the blendshapes post-processor on demand and maps the coefficients to
  /// a coarse facial movement label. Returns null when blendshapes are
  /// unavailable.
  String? _resolveMovementLabel(FaceMeshResult? result) {
    final FaceBlendshapesProcessor? processor = _blendshapesProcessor;
    // Blendshapes need the 478-landmark (iris) result; skip in base mesh mode.
    // Both the iris and attention modes provide it.
    if (result == null || processor == null || !result.hasIris) {
      return null;
    }
    final FaceBlendshapes? blendshapes = processor.process(result);
    if (blendshapes == null) {
      return null; // no face in this frame
    }
    return _detectMovement(blendshapes);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Desktop windows report inactive whenever they lose focus; only mobile
    // camera sources need the release/reacquire dance.
    if (!_frameSource.supportsLifecyclePause || !_frameSource.isReady) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      void reset() {
        _isCameraActive = false;
        _isDetectionActive = false;
        _isMeshActive = false;
        _clearMesh();
        _resetInference();
        _clearDetections();
        _clearInferenceFps();
      }

      _update(reset);
      _frameSource.stop();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _frameSource.onFrame = null;
    _frameSource.removeListener(_onFrameSourceChanged);
    _frameSource.dispose();
    _resetInference();
    _faceMeshInferencePipeline.close();
    _blendshapesProcessor?.close();
    _isolatePipeline?.close();
    _controlsScrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isCameraAvailable = _isCameraActive && _frameSource.isReady;

    return Scaffold(
      appBar: AppBar(
        title: const Text('mediapipe_face_mesh'),
        titleTextStyle: const TextStyle(color: Colors.black, fontSize: 16),
        centerTitle: true,
      ),
      body: SafeArea(
        child: _errorMessage != null
            ? _buildErrorView()
            : _isInitializing
            ? const Center(child: CircularProgressIndicator())
            : Column(
                children: [
                  Center(child: _buildCameraPreview(isCameraAvailable)),
                  SizedBox(height: 10),
                  Expanded(
                    child: Scrollbar(
                      controller: _controlsScrollController,
                      thumbVisibility: true,
                      child: SingleChildScrollView(
                        controller: _controlsScrollController,
                        child: Column(
                          children: [
                            ..._buildSourceSelectors(),
                            _buildModelSelector(),
                            _buildMeshModelSelector(),
                            _buildMultiFaceSwitch(),
                            _buildCameraOptionsPanel(),
                            _buildImageProcessOptionsPanel(),
                            _buildControlButtons(),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Text(
          _errorMessage ?? 'Unknown error',
          style: const TextStyle(color: Colors.red),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }

  Widget _buildCameraPreview(bool isCameraAvailable) {
    final nativeAspectRatio = _frameSource.nativeAspectRatio;
    final displayAspectRatio = _frameSource.displayAspectRatio;
    // Results are drawn as-is (no compensation for the Image process
    // options), so an input-side rotation or mirror is visible on screen:
    // the mesh draws where the transformed coordinates say it is.
    final mirror = _frameSource.mirrorHorizontal;
    final fpsText =
        'Infer: ${_inferenceFps > 0 ? _inferenceFps.toStringAsFixed(1) : '--'} fps';

    return Builder(
      builder: (context) {
        final Size screen = MediaQuery.of(context).size;
        // Cap by height too so wide desktop windows keep room for controls.
        final displayWidth = math.min(
          screen.width * 0.9,
          screen.height * 0.55 * displayAspectRatio,
        );
        // Inner SizedBox keeps the camera's native ratio so it renders correctly.
        final nativeHeight = displayWidth / nativeAspectRatio;

        return SizedBox(
          width: displayWidth,
          child: AspectRatio(
            aspectRatio: displayAspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Camera feed clipped to display ratio. The Camera options
                // rotate/mirror the composited preview and overlays as one
                // layer, so they cannot drift apart.
                ClipRect(
                  child: Transform.flip(
                    flipX: _userMirror,
                    flipY: _userFlipVertical,
                    child: RotatedBox(
                      quarterTurns: _userRotationDegrees ~/ 90,
                      child: FittedBox(
                        fit: BoxFit.cover,
                        child: SizedBox(
                          width: displayWidth,
                          height: nativeHeight,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              if (isCameraAvailable)
                                _frameSource.buildPreview()
                              else
                                Container(
                                  color: Colors.black12,
                                  alignment: Alignment.center,
                                  child: const Text(
                                    'Press Start Cam',
                                    style: TextStyle(color: Colors.black54),
                                  ),
                                ),
                              // Detector frames draw the detector's ROI,
                              // tracked frames the ROI the mesh used.
                              if (isCameraAvailable && _inference != null)
                                RepaintBoundary(
                                  child: CustomPaint(
                                    painter: FaceDetectionPainter.fromInference(
                                      _inference!,
                                      // The movement from the blendshapes,
                                      // as in the multi-face labels.
                                      label: _movementLabel,
                                      mirrorHorizontal: mirror,
                                      showConfidence: false,
                                      showFaceBox: false,
                                    ),
                                  ),
                                ),
                              if (isCameraAvailable && _multiInference != null)
                                RepaintBoundary(
                                  child: CustomPaint(
                                    painter:
                                        FaceDetectionPainter.fromMultiInference(
                                          _multiInference!,
                                          labelOf: (TrackedFaceMesh face) =>
                                              _multiFaceLabels[face.trackId] ??
                                              '#${face.trackId}',
                                          mirrorHorizontal: mirror,
                                          showConfidence: false,
                                          showFaceBox: false,
                                          // Tracked faces carry their own
                                          // ROI. Detector ROIs only matter
                                          // while the mesh is off.
                                          showRoiBox: !_isMeshActive,
                                        ),
                                  ),
                                ),
                              if (isCameraAvailable &&
                                  _isMeshActive &&
                                  _inference?.meshResult != null)
                                RepaintBoundary(
                                  child: IgnorePointer(
                                    child: CustomPaint(
                                      painter: FaceMeshPainter.fromInference(
                                        _inference!,
                                        irisDotRadius: 2,
                                        scaleWithFace: true,
                                        mirrorHorizontal: mirror,
                                      ),
                                    ),
                                  ),
                                ),
                              if (isCameraAvailable &&
                                  _isMeshActive &&
                                  (_multiInference?.faces.isNotEmpty ?? false))
                                RepaintBoundary(
                                  child: IgnorePointer(
                                    child: CustomPaint(
                                      painter:
                                          FaceMeshPainter.fromMultiInference(
                                            _multiInference!,
                                            irisDotRadius: 2,
                                            scaleWithFace: true,
                                            mirrorHorizontal: mirror,
                                          ),
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                // Chips outside ClipRect so they're always visible
                if (isCameraAvailable)
                  Positioned(top: 12, right: 12, child: _infoChip(fpsText)),
                if (_geometryText != null)
                  Positioned(
                    top: 12,
                    left: 12,
                    child: _infoChip(_geometryText!),
                  ),
                Positioned(
                  bottom: 12,
                  left: 12,
                  child: _infoChip(_trackingChipText()),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  String _trackingChipText() {
    final int trackedFaces = _multiInference?.faces.length ?? 0;
    if (_isMultiFaceActive && trackedFaces > 0) {
      return 'Tracking $trackedFaces/$_maxMeshFaces';
    }
    if (!_isMultiFaceActive && _inference?.detectorRan == false) {
      return 'Tracking';
    }
    final FaceDetectionResult? detections =
        _inference?.detectionResult ?? _multiInference?.detectionResult;
    return 'Faces: ${detections?.detections.length ?? 0}';
  }

  String _geometryTextOf(FaceMeshResult result) {
    try {
      final geometry = result.estimateGeometry();
      final pose = geometry.headPose;
      final measurements = geometry.measurements;
      final double innerEyePixels = result.distancePixels(133, 362);
      final StringBuffer buf = StringBuffer(
        'Yaw ${pose.yawDegrees.toStringAsFixed(0)}°  '
        'Pitch ${pose.pitchDegrees.toStringAsFixed(0)}°  '
        'Roll ${pose.rollDegrees.toStringAsFixed(0)}°\n',
      );
      final ipd = measurements.interpupillaryDistance;
      if (ipd != null) {
        buf.write('IPD ${ipd.valueCm.toStringAsFixed(1)}cm  ');
      }
      buf.write(
        'Inner eye ${measurements.eyeInnerDistance.valueCm.toStringAsFixed(1)}cm\n'
        'Inner eye ${innerEyePixels.toStringAsFixed(0)}px',
      );
      return buf.toString();
    } on Object {
      return 'Geometry unavailable';
    }
  }

  /// Maps the 52 blendshape coefficients to a coarse facial movement label.
  ///
  /// Thresholds are illustrative starting points; tune per camera and lighting.
  String _detectMovement(FaceBlendshapes blendshapes) {
    double v(FaceBlendshape shape) => blendshapes[shape];
    final double smile =
        (v(FaceBlendshape.mouthSmileLeft) + v(FaceBlendshape.mouthSmileRight)) /
        2;
    final double blink = math.max(
      v(FaceBlendshape.eyeBlinkLeft),
      v(FaceBlendshape.eyeBlinkRight),
    );

    if (blink > 0.45) {
      return 'Blink';
    }
    if (v(FaceBlendshape.jawOpen) > 0.35) {
      return 'Mouth open';
    }
    if (smile > 0.4) {
      return 'Smile';
    }
    return 'Neutral';
  }

  Widget _infoChip(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  static const TextStyle _selectorTextStyle = TextStyle(
    fontSize: 13,
    fontWeight: FontWeight.w600,
    color: Colors.black87,
  );

  InputDecoration _selectorDecoration(String label) {
    OutlineInputBorder border(Color color, [double width = 1]) =>
        OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: color, width: width),
        );
    return InputDecoration(
      labelText: label,
      isDense: true,
      filled: true,
      fillColor: Colors.black.withValues(alpha: 0.035),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      labelStyle: const TextStyle(fontSize: 12, color: Colors.black54),
      floatingLabelStyle: const TextStyle(fontSize: 12.5),
      border: border(Colors.black12),
      enabledBorder: border(Colors.black12),
      focusedBorder: border(Colors.black38, 1.4),
    );
  }

  /// Source-provided chip filters and dropdowns (UVC format filter, device
  /// and camera mode on Windows), styled like the model selectors below.
  /// Empty for the mobile camera source.
  List<Widget> _buildSourceSelectors() {
    return [
      for (final FrameSourceTagFilter filter in _frameSource.tagFilters)
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
          child: Align(
            alignment: Alignment.centerLeft,
            child: Wrap(
              spacing: 8,
              children: [
                for (var i = 0; i < filter.options.length; i++)
                  ChoiceChip(
                    label: Text(filter.options[i]),
                    selected: filter.selectedIndex == i,
                    onSelected: _isCameraBusy
                        ? null
                        : (_) => filter.onSelect(i),
                  ),
              ],
            ),
          ),
        ),
      for (final FrameSourceSelector selector in _frameSource.selectors)
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
          child: DropdownButtonFormField<int>(
            // initialValue needs Flutter 3.35; the package supports 3.32.
            // ignore: deprecated_member_use
            value: selector.selectedIndex >= 0 ? selector.selectedIndex : null,
            isDense: true,
            isExpanded: true,
            borderRadius: BorderRadius.circular(12),
            style: _selectorTextStyle,
            icon: const Icon(Icons.expand_more_rounded, size: 20),
            decoration: _selectorDecoration(selector.label),
            items: [
              for (var i = 0; i < selector.options.length; i++)
                DropdownMenuItem<int>(
                  value: i,
                  child: Text(
                    selector.options[i],
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
            ],
            onChanged: _isCameraBusy
                ? null
                : (index) {
                    if (index == null || index == selector.selectedIndex) {
                      return;
                    }
                    selector.onSelect(index);
                  },
          ),
        ),
    ];
  }

  Widget _buildModelSelector() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
      child: DropdownButtonFormField<FaceDetectionModel>(
        // ignore: deprecated_member_use
        value: _detectionModel,
        isDense: true,
        borderRadius: BorderRadius.circular(12),
        style: _selectorTextStyle,
        icon: const Icon(Icons.expand_more_rounded, size: 20),
        decoration: _selectorDecoration('Detection Model'),
        items: const [
          DropdownMenuItem<FaceDetectionModel>(
            value: FaceDetectionModel.shortRange,
            child: Text('Short-range'),
          ),
          DropdownMenuItem<FaceDetectionModel>(
            value: FaceDetectionModel.fullRange,
            child: Text('Full-range (dense)'),
          ),
          DropdownMenuItem<FaceDetectionModel>(
            value: FaceDetectionModel.fullRangeSparse,
            child: Text('Full-range (sparse)'),
          ),
        ],
        onChanged: (value) {
          if (value == null || value == _detectionModel) {
            return;
          }
          _changeDetectionModel(value);
        },
      ),
    );
  }

  Widget _buildMeshModelSelector() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
      child: DropdownButtonFormField<_MeshMode>(
        // ignore: deprecated_member_use
        value: _meshMode,
        isDense: true,
        borderRadius: BorderRadius.circular(12),
        style: _selectorTextStyle,
        icon: const Icon(Icons.expand_more_rounded, size: 20),
        decoration: _selectorDecoration('Mesh Model'),
        items: [
          for (final _MeshMode mode in _MeshMode.values)
            DropdownMenuItem<_MeshMode>(value: mode, child: Text(mode.label)),
        ],
        onChanged: _isCameraBusy
            ? null
            : (value) {
                if (value == null) return;
                _changeMeshMode(value);
              },
      ),
    );
  }

  Widget _buildControlButtons() {
    final isControllerReady = _frameSource.isReady;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _isCameraBusy ? null : _toggleCamera,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isCameraActive
                        ? Colors.redAccent
                        : Colors.greenAccent,
                    foregroundColor: Colors.black,
                  ),
                  icon: Icon(
                    _isCameraActive ? Icons.stop : Icons.videocam,
                    color: Colors.black,
                  ),
                  label: Text(_isCameraActive ? 'Stop Cam' : 'Start Cam'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed:
                      (!_isCameraActive || _isCameraBusy || !isControllerReady)
                      ? null
                      : _toggleDetection,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isDetectionActive
                        ? Colors.orangeAccent
                        : Colors.blueAccent,
                    foregroundColor: Colors.black,
                  ),
                  icon: Icon(
                    _isDetectionActive ? Icons.pause : Icons.play_arrow,
                    color: Colors.black,
                  ),
                  label: Text(
                    _isDetectionActive ? 'Stop Detect' : 'Start Detect',
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed:
                      (!_isCameraActive ||
                          _isCameraBusy ||
                          !isControllerReady ||
                          !_isDetectionActive)
                      ? null
                      : _toggleMesh,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isMeshActive
                        ? Colors.purpleAccent
                        : Colors.purple,
                    foregroundColor: Colors.black,
                  ),
                  icon: Icon(
                    _isMeshActive ? Icons.stop_circle : Icons.blur_on,
                    color: Colors.black,
                  ),
                  label: Text(_isMeshActive ? 'Stop Mesh' : 'Start Mesh'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMultiFaceSwitch() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
      // Runs the mesh model on every detected face (multi-face) instead of a
      // single face. Orthogonal to the Mesh Model choice above.
      child: _buildModeSwitch(
        icon: Icons.groups,
        label: 'Multi-face mesh',
        value: _isMultiFaceActive,
        onChanged: _isCameraBusy ? null : (_) => _toggleMultiFace(),
      ),
    );
  }

  /// Camera-related controls, collapsed by default so the main controls
  /// stay short.
  Widget _buildCameraOptionsPanel() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
      child: ExpansionTile(
        shape: RoundedRectangleBorder(
          side: const BorderSide(color: Colors.black12),
          borderRadius: BorderRadius.circular(12),
        ),
        collapsedShape: RoundedRectangleBorder(
          side: const BorderSide(color: Colors.black12),
          borderRadius: BorderRadius.circular(12),
        ),
        backgroundColor: Colors.black.withValues(alpha: 0.035),
        collapsedBackgroundColor: Colors.black.withValues(alpha: 0.035),
        tilePadding: const EdgeInsets.symmetric(horizontal: 14),
        childrenPadding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
        leading: const Icon(Icons.tune, size: 18, color: Colors.black54),
        title: const Text(
          'Camera options',
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
        ),
        children: [
          _buildCameraSwitchControl(),
          const SizedBox(height: 8),
          _buildRotationControl(
            label: 'Rotate preview',
            tooltip: 'Rotate the preview by 90°',
            degrees: _userRotationDegrees,
            onRotate: () => setState(
              () => _userRotationDegrees = (_userRotationDegrees + 90) % 360,
            ),
          ),
          const SizedBox(height: 8),
          _buildModeSwitch(
            icon: Icons.flip,
            label: 'Mirror preview',
            value: _userMirror,
            onChanged: (_) => setState(() => _userMirror = !_userMirror),
          ),
          const SizedBox(height: 8),
          _buildModeSwitch(
            icon: Icons.swap_vert,
            label: 'Flip vertical preview',
            value: _userFlipVertical,
            onChanged: (_) =>
                setState(() => _userFlipVertical = !_userFlipVertical),
          ),
        ],
      ),
    );
  }

  /// Input-side transforms: unlike the display-only Camera options, these
  /// change what the processor receives and the coordinate space of its
  /// results; the demo draws results as-is, so the effect is visible.
  Widget _buildImageProcessOptionsPanel() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
      child: ExpansionTile(
        shape: RoundedRectangleBorder(
          side: const BorderSide(color: Colors.black12),
          borderRadius: BorderRadius.circular(12),
        ),
        collapsedShape: RoundedRectangleBorder(
          side: const BorderSide(color: Colors.black12),
          borderRadius: BorderRadius.circular(12),
        ),
        backgroundColor: Colors.black.withValues(alpha: 0.035),
        collapsedBackgroundColor: Colors.black.withValues(alpha: 0.035),
        tilePadding: const EdgeInsets.symmetric(horizontal: 14),
        childrenPadding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
        leading: const Icon(Icons.memory, size: 18, color: Colors.black54),
        title: const Text(
          'Image process options',
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
        ),
        children: [
          _buildRotationControl(
            label: 'Rotate input',
            tooltip: 'Rotate the pipeline input by 90°',
            degrees: _inputRotationDegrees,
            onRotate: _isCameraBusy
                ? null
                : () => setState(
                    () => _inputRotationDegrees =
                        (_inputRotationDegrees + 90) % 360,
                  ),
          ),
          const SizedBox(height: 8),
          _buildModeSwitch(
            icon: Icons.compare_arrows,
            label: 'Mirror input',
            value: _inputMirror,
            onChanged: _isCameraBusy
                ? null
                : (_) => setState(() => _inputMirror = !_inputMirror),
          ),
          const SizedBox(height: 8),
          _buildModeSwitch(
            icon: Icons.alt_route,
            label: 'Run inference in isolate',
            value: _runInIsolate,
            onChanged: _isCameraBusy ? null : (_) => _toggleRunInIsolate(),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraSwitchControl() {
    final bool enabled =
        _frameSource.canSwitch &&
        !_isChangingCamera &&
        !_isCameraBusy &&
        _isCameraActive &&
        _frameSource.isReady;
    return Container(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.035),
        border: Border.all(color: Colors.black12),
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.only(left: 14, right: 4),
      child: Row(
        children: [
          const Icon(Icons.cameraswitch, size: 18, color: Colors.black54),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              'Switch camera',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: Colors.black87,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          // front/back on mobile, the UVC device name on Windows.
          Text(
            _frameSource.activeSourceLabel,
            style: const TextStyle(fontSize: 13, color: Colors.black87),
            overflow: TextOverflow.ellipsis,
          ),
          IconButton(
            icon: const Icon(Icons.swap_horiz, size: 20),
            tooltip: 'Switch to the next camera',
            onPressed: enabled ? _switchCamera : null,
          ),
        ],
      ),
    );
  }

  Widget _buildRotationControl({
    required String label,
    required String tooltip,
    required int degrees,
    required VoidCallback? onRotate,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.035),
        border: Border.all(color: Colors.black12),
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.only(left: 14, right: 4),
      child: Row(
        children: [
          const Icon(
            Icons.screen_rotation_alt,
            size: 18,
            color: Colors.black54,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: Colors.black87,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          Text(
            '$degrees°',
            style: const TextStyle(fontSize: 13, color: Colors.black87),
          ),
          IconButton(
            icon: const Icon(Icons.rotate_right, size: 20),
            tooltip: tooltip,
            onPressed: onRotate,
          ),
        ],
      ),
    );
  }

  Widget _buildModeSwitch({
    required IconData icon,
    required String label,
    required bool value,
    required ValueChanged<bool>? onChanged,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.035),
        border: Border.all(color: Colors.black12),
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.only(left: 14),
      child: Row(
        children: [
          Icon(icon, size: 18, color: Colors.black54),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: Colors.black87,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          Transform.scale(
            scale: 0.78,
            child: Switch(
              value: value,
              onChanged: onChanged,
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _toggleCamera() async {
    if (_isCameraBusy) {
      return;
    }
    if (_isCameraActive) {
      await _stopCamera();
    } else {
      await _startCamera();
    }
  }

  Future<void> _startCamera() async {
    if (_isCameraBusy || _isCameraActive) {
      return;
    }
    if (mounted) {
      setState(() {
        _isCameraBusy = true;
        _errorMessage = null;
        _isDetectionActive = false;
        _isMeshActive = false;
        _clearMesh();
        _resetInference();
        _clearDetections();
      });
    }
    try {
      final initialized = await _startFrameSource();
      _update(() => _isCameraActive = initialized);
    } finally {
      _update(() => _isCameraBusy = false);
    }
  }

  Future<void> _stopCamera() async {
    void reset() {
      _isCameraActive = false;
      _isDetectionActive = false;
      _isMeshActive = false;
      _clearMesh();
      _resetInference();
      _clearInferenceFps();
      _clearDetections();
    }

    if (!_isCameraActive) {
      _update(reset);
      return;
    }
    _update(() {
      _isCameraBusy = true;
      reset();
    });

    try {
      await _frameSource.stop();
    } catch (error) {
      _errorMessage ??= '$error';
    } finally {
      _update(() => _isCameraBusy = false);
    }
  }

  Future<void> _switchCamera() async {
    if (!_frameSource.canSwitch ||
        _isChangingCamera ||
        _isCameraBusy ||
        !_isCameraActive) {
      return;
    }

    _update(() => _isChangingCamera = true);

    _resetInference();
    _clearDetections();
    _clearInferenceFps();

    try {
      final initialized = await _frameSource.switchSource();
      if (!initialized) {
        _errorMessage ??= _frameSource.lastError;
        _update(() => _isCameraActive = false);
      }
    } finally {
      _update(() => _isChangingCamera = false);
    }
  }

  void _handleSourceFrame(DemoFrame frame) {
    // One request in flight. Requests queue in the worker, so a frame that
    // arrives while one is pending is dropped. The synchronous mode needs
    // no gate. Its call returns before this callback does.
    if (_isolatePipeline?.isBusy ?? false) {
      return;
    }
    if (!_frameSource.isReady || !_isCameraActive || !_isDetectionActive) {
      return;
    }
    try {
      _pushFrameToDetectionStage(frame);
    } catch (error) {
      _update(() => _errorMessage ??= '$error');
    }
  }

  bool _isDetectionStageActive() {
    return mounted && _isCameraActive && _isDetectionActive;
  }

  void _pushFrameToDetectionStage(DemoFrame frame) {
    final rotationCompensation = _frameSource.rotationCompensationDegrees;
    if (rotationCompensation == null) {
      return;
    }
    final int effectiveRotation =
        (rotationCompensation + _inputRotationDegrees) % 360;
    final FaceMeshFrame? input = frame.nv21 ?? frame.image;
    if (input == null) {
      return;
    }
    if (_runInIsolate) {
      unawaited(
        _pushFrameToIsolate(
          input,
          rotationDegrees: effectiveRotation,
          mirrorHorizontal: _inputMirror,
        ),
      );
      return;
    }
    if (_noteInputRotation(effectiveRotation)) {
      _faceMeshInferencePipeline.resetTracking();
    }
    try {
      if (_isMultiFaceActive) {
        _handleMultiInferenceResult(
          _faceMeshInferencePipeline.processMultiFace(
            input,
            maxMeshFaces: _maxMeshFaces,
            runMesh: _isMeshActive,
            rotationDegrees: effectiveRotation,
            mirrorHorizontal: _inputMirror,
          ),
        );
      } else {
        _handleInferenceResult(
          _faceMeshInferencePipeline.process(
            input,
            runMesh: _isMeshActive,
            rotationDegrees: effectiveRotation,
            mirrorHorizontal: _inputMirror,
          ),
        );
      }
    } catch (error) {
      _handleInferenceError(error);
    }
  }

  /// Same as the synchronous path, but the result arrives later. One
  /// request in flight (see _handleSourceFrame), stale replies dropped.
  Future<void> _pushFrameToIsolate(
    FaceMeshFrame input, {
    required int rotationDegrees,
    required bool mirrorHorizontal,
  }) async {
    final FaceMeshIsolatePipeline? pipeline = _isolatePipeline;
    if (pipeline == null) {
      return;
    }
    if (pipeline.isClosed) {
      // The worker exited. The error that killed it was already reported.
      _isolatePipeline = null;
      return;
    }
    // Replies from a worker that was replaced meanwhile are dropped.
    bool isCurrent() => identical(pipeline, _isolatePipeline);

    try {
      if (_noteInputRotation(rotationDegrees)) {
        await pipeline.resetTracking();
      }
      if (_isMultiFaceActive) {
        final FaceMeshMultiInferenceResult result = await pipeline
            .processMultiFace(
              input,
              maxMeshFaces: _maxMeshFaces,
              runMesh: _isMeshActive,
              rotationDegrees: rotationDegrees,
              mirrorHorizontal: mirrorHorizontal,
            );
        if (isCurrent()) {
          _handleMultiInferenceResult(result);
        }
      } else {
        final FaceMeshInferenceResult result = await pipeline.process(
          input,
          runMesh: _isMeshActive,
          rotationDegrees: rotationDegrees,
          mirrorHorizontal: mirrorHorizontal,
        );
        if (isCurrent()) {
          _handleInferenceResult(result);
        }
      }
    } catch (error) {
      if (isCurrent()) {
        _handleInferenceError(error);
      }
    }
  }

  Future<void> _toggleDetection() async {
    if (!_frameSource.isReady || _isCameraBusy) {
      return;
    }

    if (_isDetectionActive) {
      if (mounted) {
        setState(() {
          _isDetectionActive = false;
          _isMeshActive = false;
          _clearMesh();
          _resetInference();
          _clearDetections();
        });
      }
      return;
    }

    try {
      await _frameSource.ensureFrames();
      _update(() {
        _isDetectionActive = true;
        _clearDetections();
      });
    } catch (error) {
      if (mounted) {
        setState(() => _errorMessage = 'Detection start error: $error');
      }
    }
  }

  Future<void> _toggleMesh() async {
    if (_isCameraBusy || !_frameSource.isReady) {
      return;
    }

    if (!_isDetectionActive) {
      if (mounted) {
        setState(
          () => _errorMessage ??= 'Start Detect first to get a face ROI.',
        );
      }
      return;
    }

    if (_isMeshActive) {
      if (mounted) {
        setState(() {
          _isMeshActive = false;
          _clearMesh();
        });
      }
      return;
    }

    _update(() {
      _isMeshActive = true;
      _clearMesh();
    });
  }

  Future<void> _changeMeshMode(_MeshMode mode) async {
    if (_isCameraBusy || mode == _meshMode) return;
    final previous = _meshMode;
    _update(() => _meshMode = mode);
    try {
      await _replaceFaceMeshProcessor(model: mode.model, iris: mode.enableIris);
    } catch (error) {
      _update(() {
        _meshMode = previous;
        _errorMessage = 'Mesh model change error: $error';
      });
    }
  }

  /// The same pipeline serves both flows.
  void _toggleMultiFace() {
    if (_isCameraBusy) return;
    _resetInference();
    _clearMesh();
    _clearDetections();
    setState(() => _isMultiFaceActive = !_isMultiFaceActive);
  }

  /// Swaps the mesh processor and rebuilds the pipeline.
  Future<void> _replaceFaceMeshProcessor({
    required FaceMeshModel model,
    required bool iris,
  }) async {
    final newProcessor = await _createFaceMeshProcessor(
      model: model,
      iris: iris,
    );
    _resetInference();
    _clearMesh();
    _clearDetections();
    final oldProcessor = _faceMeshProcessor;
    _faceMeshProcessor = newProcessor;
    _faceMeshInferencePipeline = FaceMeshInferencePipeline(
      detector: _faceDetectorProcessor,
      mesh: _faceMeshProcessor,
      landmarkSmoothing: _landmarkSmoothing,
    );
    oldProcessor.close();
    unawaited(_restartIsolatePipeline());
  }
}
