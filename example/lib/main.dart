import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'inference_config.dart';
import 'sources/camera_frame_source.dart';
import 'sources/frame_source.dart';
import 'sources/uvc_frame_source.dart';
import 'utils/result_text.dart';
import 'widgets/control_bar.dart';
import 'widgets/error_banner.dart';
import 'widgets/option_tiles.dart';
import 'widgets/preview_view.dart';
import 'widgets/source_selectors.dart';

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

  /// Whether frames are handed to the pipeline. On after the camera starts;
  /// the Stop Detect button turns it off.
  bool _isDetectionActive = false;

  /// Whether the mesh runs on detected faces. A preference that survives
  /// camera stops.
  bool _isMeshActive = true;

  /// Run the pipeline in a worker isolate (FaceMeshIsolatePipeline) instead
  /// of on the UI isolate. On by default. Rebuilt whenever a model changes.
  bool _runInIsolate = true;
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

  /// False until _initialize built the processors. The camera stays off and
  /// dispose skips them when it failed.
  bool _isPipelineReady = false;

  /// Rotation of the last frame handed to inference. Null until the first
  /// frame after _resetInference, which marks a new input source.
  int? _lastInferenceRotation;
  FaceDetectionModel _detectionModel = FaceDetectionModel.shortRange;
  MeshMode _meshMode = MeshMode.faceMeshV2;
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
    FaceDetectorProcessor? detector;
    FaceMeshProcessor? mesh;
    try {
      detector = await createFaceDetectorProcessor(_detectionModel);
      mesh = await createFaceMeshProcessor(
        model: _meshMode.model,
        iris: _meshMode.enableIris,
      );
      // Create the blendshapes processor once (it loads the model), then run it
      // on each mesh result below (the mesh must include iris landmarks).
      _blendshapesProcessor = await FaceBlendshapesProcessor.create(
        delegate: preferredDelegate,
      );
      _faceDetectorProcessor = detector;
      _faceMeshProcessor = mesh;
      _faceMeshInferencePipeline = FaceMeshInferencePipeline(
        detector: detector,
        mesh: mesh,
        landmarkSmoothing: landmarkSmoothing,
      );
      _isPipelineReady = true;
      // Spawns the worker when isolate mode is on, so Start Cam has one.
      await _restartIsolatePipeline();
    } catch (error) {
      _errorMessage = '$error';
      if (!_isPipelineReady) {
        detector?.close();
        mesh?.close();
      }
    } finally {
      _update(() => _isInitializing = false);
    }
  }

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
      final newFaceDetectorProcessor = await createFaceDetectorProcessor(
        _detectionModel,
      );
      _resetInference();
      _clearDetections();
      final oldProcessor = _faceDetectorProcessor;
      _faceDetectorProcessor = newFaceDetectorProcessor;
      _faceMeshInferencePipeline = FaceMeshInferencePipeline(
        detector: newFaceDetectorProcessor,
        mesh: _faceMeshProcessor,
        landmarkSmoothing: landmarkSmoothing,
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

  IsolatePipelineArgs _currentIsolateArgs() =>
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
    final IsolatePipelineArgs args = _currentIsolateArgs();
    try {
      final FaceMeshIsolatePipeline pipeline =
          await FaceMeshIsolatePipeline.spawn(createPipelineInWorker, args);
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
        ? geometryTextOf(meshResult)
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
    return detectMovement(blendshapes);
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
    if (_isPipelineReady) {
      _faceMeshInferencePipeline.close();
    }
    _blendshapesProcessor?.close();
    _isolatePipeline?.close();
    _controlsScrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bool isCameraAvailable = _isCameraActive && _frameSource.isReady;
    final bool canSwitchCamera =
        !_isChangingCamera && !_isCameraBusy && _frameSource.isReady;
    final bool canToggleDetect =
        _isCameraActive && !_isCameraBusy && _frameSource.isReady;

    return Scaffold(
      appBar: AppBar(
        title: const Text('mediapipe_face_mesh'),
        titleTextStyle: const TextStyle(color: Colors.black, fontSize: 16),
        centerTitle: true,
      ),
      body: SafeArea(
        child: _isInitializing
            ? const Center(child: CircularProgressIndicator())
            : Column(
                children: [
                  if (_errorMessage != null)
                    ErrorBanner(
                      message: _errorMessage!,
                      onDismiss: () => setState(() => _errorMessage = null),
                    ),
                  Center(
                    child: PreviewView(
                      frameSource: _frameSource,
                      isCameraAvailable: isCameraAvailable,
                      inference: _inference,
                      multiInference: _multiInference,
                      multiFaceLabels: _multiFaceLabels,
                      movementLabel: _movementLabel,
                      geometryText: _geometryText,
                      inferenceFps: _inferenceFps,
                      showMesh: _isMeshActive,
                      isMultiFace: _isMultiFaceActive,
                      maxMeshFaces: maxMeshFaces,
                      rotationDegrees: _userRotationDegrees,
                      mirror: _userMirror,
                      flipVertical: _userFlipVertical,
                      onSwitchCamera: canSwitchCamera ? _switchCamera : null,
                    ),
                  ),
                  ControlBar(
                    isCameraActive: _isCameraActive,
                    isDetectionActive: _isDetectionActive,
                    onToggleCamera: _isCameraBusy || !_isPipelineReady
                        ? null
                        : _toggleCamera,
                    onToggleDetection: canToggleDetect
                        ? _toggleDetection
                        : null,
                  ),
                  Expanded(
                    child: Scrollbar(
                      controller: _controlsScrollController,
                      thumbVisibility: true,
                      child: SingleChildScrollView(
                        controller: _controlsScrollController,
                        child: Column(
                          children: [
                            SourceSelectors(
                              frameSource: _frameSource,
                              enabled: !_isCameraBusy,
                            ),
                            OptionsPanel(
                              title: 'Models',
                              initiallyExpanded: true,
                              children: [
                                _buildModelSelector(),
                                const SizedBox(height: 8),
                                _buildMeshModelSelector(),
                              ],
                            ),
                            _buildMeshOptionsPanel(),
                            _buildCameraOptionsPanel(),
                            _buildImageProcessOptionsPanel(),
                            const SizedBox(height: 8),
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

  Widget _buildModelSelector() {
    return LabeledDropdown<FaceDetectionModel>(
      label: 'Detection Model',
      value: _detectionModel,
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
    );
  }

  Widget _buildMeshModelSelector() {
    return LabeledDropdown<MeshMode>(
      label: 'Mesh Model',
      value: _meshMode,
      items: [
        for (final MeshMode mode in MeshMode.values)
          DropdownMenuItem<MeshMode>(value: mode, child: Text(mode.label)),
      ],
      onChanged: _isCameraBusy
          ? null
          : (value) {
              if (value == null) return;
              _changeMeshMode(value);
            },
    );
  }

  Widget _buildMeshOptionsPanel() {
    return OptionsPanel(
      title: 'Mesh options',
      initiallyExpanded: true,
      children: [
        SwitchTile(
          label: 'Face mesh',
          value: _isMeshActive,
          onChanged: _isCameraBusy ? null : (_) => _toggleMesh(),
        ),
        const SizedBox(height: 8),
        // Runs the mesh on every detected face instead of one. Orthogonal
        // to the Mesh Model choice.
        SwitchTile(
          label: 'Multi-face mesh',
          value: _isMultiFaceActive,
          onChanged: _isCameraBusy ? null : (_) => _toggleMultiFace(),
        ),
        const SizedBox(height: 8),
        SwitchTile(
          label: 'Run inference in isolate',
          value: _runInIsolate,
          onChanged: _isCameraBusy ? null : (_) => _toggleRunInIsolate(),
        ),
      ],
    );
  }

  /// Display-side transforms. Collapsed by default.
  Widget _buildCameraOptionsPanel() {
    return OptionsPanel(
      title: 'Camera options',
      children: [
        RotationTile(
          label: 'Rotate preview',
          tooltip: 'Rotate the preview by 90°',
          degrees: _userRotationDegrees,
          onRotate: () => setState(
            () => _userRotationDegrees = (_userRotationDegrees + 90) % 360,
          ),
        ),
        const SizedBox(height: 8),
        SwitchTile(
          label: 'Mirror preview',
          value: _userMirror,
          onChanged: (_) => setState(() => _userMirror = !_userMirror),
        ),
        const SizedBox(height: 8),
        SwitchTile(
          label: 'Flip vertical preview',
          value: _userFlipVertical,
          onChanged: (_) =>
              setState(() => _userFlipVertical = !_userFlipVertical),
        ),
      ],
    );
  }

  /// Input-side transforms: unlike the display-only Camera options, these
  /// change what the processor receives and the coordinate space of its
  /// results; the demo draws results as-is, so the effect is visible.
  Widget _buildImageProcessOptionsPanel() {
    return OptionsPanel(
      title: 'Image process options',
      children: [
        RotationTile(
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
        SwitchTile(
          label: 'Mirror input',
          value: _inputMirror,
          onChanged: _isCameraBusy
              ? null
              : (_) => setState(() => _inputMirror = !_inputMirror),
        ),
      ],
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
        _clearMesh();
        _resetInference();
        _clearDetections();
      });
    }
    try {
      final initialized = await _startFrameSource();
      _update(() => _isCameraActive = initialized);
      if (initialized) {
        await _startDetection();
      }
    } finally {
      _update(() => _isCameraBusy = false);
    }
  }

  Future<void> _stopCamera() async {
    void reset() {
      _isCameraActive = false;
      _isDetectionActive = false;
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
            maxMeshFaces: maxMeshFaces,
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
              maxMeshFaces: maxMeshFaces,
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
      _update(() {
        _isDetectionActive = false;
        _clearMesh();
        _resetInference();
        _clearDetections();
      });
      return;
    }
    await _startDetection();
  }

  Future<void> _startDetection() async {
    try {
      await _frameSource.ensureFrames();
      _update(() {
        _isDetectionActive = true;
        _clearDetections();
      });
    } catch (error) {
      _update(() => _errorMessage = 'Detection start error: $error');
    }
  }

  void _toggleMesh() {
    if (_isCameraBusy) {
      return;
    }
    setState(() {
      _isMeshActive = !_isMeshActive;
      _clearMesh();
    });
  }

  Future<void> _changeMeshMode(MeshMode mode) async {
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
    final newProcessor = await createFaceMeshProcessor(
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
      landmarkSmoothing: landmarkSmoothing,
    );
    oldProcessor.close();
    unawaited(_restartIsolatePipeline());
  }
}
