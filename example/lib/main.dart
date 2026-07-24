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

class _DetectionSnapshot {
  const _DetectionSnapshot({
    required this.result,
    required this.rotationDegrees,
  });

  /// Null when the detector was skipped for a landmark-tracked frame.
  final FaceDetectionResult? result;
  final int rotationDegrees;
}

/// One tracked-ROI overlay entry: the rotated ROI and an optional label
/// (the multi-face track id).
class _TrackedRoiOverlay {
  const _TrackedRoiOverlay({required this.roi, this.label});

  final NormalizedRect roi;
  final String? label;
}

/// Draws the rotated ROIs that landmark tracking used for mesh inference.
///
/// Shown while the detector is skipped, in place of the detection ROI boxes.
class _TrackedRoiPainter extends CustomPainter {
  const _TrackedRoiPainter({
    required this.overlays,
    this.mirrorHorizontal = false,
  });

  final List<_TrackedRoiOverlay> overlays;
  final bool mirrorHorizontal;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    for (final _TrackedRoiOverlay overlay in overlays) {
      _paintOverlay(canvas, size, overlay, paint);
    }
  }

  void _paintOverlay(
    Canvas canvas,
    Size size,
    _TrackedRoiOverlay overlay,
    Paint paint,
  ) {
    final NormalizedRect roi = overlay.roi;
    final double centerX = roi.xCenter * size.width;
    final double centerY = roi.yCenter * size.height;
    final double width = roi.width * size.width;
    final double height = roi.height * size.height;
    final double cosR = math.cos(roi.rotation);
    final double sinR = math.sin(roi.rotation);
    final List<Offset> corners =
        <Offset>[
          Offset(-width * 0.5, -height * 0.5),
          Offset(width * 0.5, -height * 0.5),
          Offset(width * 0.5, height * 0.5),
          Offset(-width * 0.5, height * 0.5),
        ].map((Offset corner) {
          double x = centerX + cosR * corner.dx - sinR * corner.dy;
          final double y = centerY + sinR * corner.dx + cosR * corner.dy;
          if (mirrorHorizontal) {
            x = size.width - x;
          }
          return Offset(x, y);
        }).toList();

    final Path path = Path()
      ..moveTo(corners[0].dx, corners[0].dy)
      ..lineTo(corners[1].dx, corners[1].dy)
      ..lineTo(corners[2].dx, corners[2].dy)
      ..lineTo(corners[3].dx, corners[3].dy)
      ..close();
    canvas.drawPath(path, paint);

    final String? label = overlay.label;
    if (label == null) {
      return;
    }
    double minX = corners.first.dx;
    double minY = corners.first.dy;
    for (final Offset corner in corners.skip(1)) {
      minX = math.min(minX, corner.dx);
      minY = math.min(minY, corner.dy);
    }
    final TextPainter textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.black,
          fontSize: 14,
          fontWeight: FontWeight.w700,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final Rect background = Rect.fromLTWH(
      minX,
      math.max(0, minY - textPainter.height - 4),
      textPainter.width + 8,
      textPainter.height + 4,
    );
    canvas.drawRect(
      background,
      Paint()..color = Colors.cyanAccent.withValues(alpha: 0.85),
    );
    textPainter.paint(canvas, Offset(background.left + 4, background.top + 2));
  }

  @override
  bool shouldRepaint(covariant _TrackedRoiPainter oldDelegate) {
    return oldDelegate.overlays != overlays ||
        oldDelegate.mirrorHorizontal != mirrorHorizontal;
  }
}

class _StageInputControllers {
  StreamController<FaceMeshNv21Image>? nv21Controller;
  StreamController<FaceMeshImage>? bgraController;

  void close() {
    nv21Controller?.close();
    bgraController?.close();
    nv21Controller = null;
    bgraController = null;
  }
}

class MediaPipeFacePage extends StatefulWidget {
  const MediaPipeFacePage({super.key, required this.frameSource});

  final DemoFrameSource frameSource;

  @override
  State<MediaPipeFacePage> createState() => _MediaPipeFacePageState();
}

/// Face mesh model selection: base mesh, base + iris two-pass, or the unified
/// attention model. Iris and attention are mutually exclusive by construction,
/// so a single choice avoids ambiguous combinations.
enum _MeshMode {
  base('Mesh (468)'),
  iris('Mesh (468) + Iris (10)'),
  attention('Attention Mesh (478)');

  const _MeshMode(this.label);

  final String label;

  bool get enableIris => this == _MeshMode.iris;
  bool get enableAttention => this == _MeshMode.attention;

  /// Whether the result includes the 478-landmark iris set (required by
  /// blendshapes). Both iris and attention modes produce it.
  bool get has478 => this != _MeshMode.base;
}

class _MediaPipeFacePageState extends State<MediaPipeFacePage>
    with WidgetsBindingObserver {
  static const String _shortRangeModel = 'short_range';
  static const String _fullRangeDenseModel = 'full_range_dense';
  static const String _fullRangeSparseModel = 'full_range_sparse';

  DemoFrameSource get _frameSource => widget.frameSource;
  String? _errorMessage;
  bool _isInitializing = true;
  bool _isCameraActive = false;
  bool _isCameraBusy = false;
  bool _isChangingCamera = false;
  bool _isDetectionActive = false;
  bool _isMeshActive = false;
  bool _isProcessingFrame = false;
  static const Duration _cameraFpsUpdateInterval = Duration(milliseconds: 200);
  double _cameraFps = 0;
  DateTime? _lastCameraFrameTime;
  DateTime? _lastCameraFpsUpdateTime;
  FaceDetectionResult? _detectionResult;

  /// ROIs reported by landmark tracking while the detector is skipped —
  /// one entry in single-face mode, one per tracked face in multi mode.
  List<_TrackedRoiOverlay> _trackedRoiOverlays = const <_TrackedRoiOverlay>[];

  /// Faces reported by the multi-face tracking flow.
  List<TrackedFaceMesh> _multiFaces = const <TrackedFaceMesh>[];
  FaceMeshResult? _meshResult;
  int? _meshRotationCompensation;
  String? _movementLabel;
  FaceBlendshapesProcessor? _blendshapesProcessor;
  late FaceDetectorProcessor _faceDetectorProcessor;
  late FaceMeshProcessor _faceMeshProcessor;
  late FaceMeshInferencePipeline _faceMeshInferencePipeline;
  late FaceMeshInferenceStreamProcessor _faceMeshInferenceStreamProcessor;
  final _inferenceStageInput = _StageInputControllers();
  StreamSubscription<Object>? _inferenceStreamSubscription;
  int? _inferenceStreamRotation;
  String _selectedModel = _shortRangeModel;
  // The attention model needs MediaPipe custom TFLite ops, which the bundled
  // Windows runtime does not include — default to the iris two-pass there.
  _MeshMode _meshMode = Platform.isWindows
      ? _MeshMode.iris
      : _MeshMode.attention;
  bool _isMultiFaceActive = false;
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
      _faceDetectorProcessor = await _createFaceDetectorProcessor();

      final faceMeshProcessor = await _createFaceMeshProcessor(
        multi: _isMultiFaceActive,
        iris: _meshMode.enableIris,
        attention: _meshMode.enableAttention,
      );
      // Create the blendshapes processor once (it loads the model), then run it
      // on each mesh result below (the mesh must include iris landmarks).
      _blendshapesProcessor = await FaceBlendshapesProcessor.create(
        delegate: FaceMeshDelegate.xnnpack,
      );
      final inferencePipeline = FaceMeshInferencePipeline(
        detector: _faceDetectorProcessor,
        mesh: faceMeshProcessor,
      );
      final inferenceStreamProcessor = FaceMeshInferenceStreamProcessor(
        inferencePipeline,
      );
      if (mounted) {
        setState(() {
          _faceMeshProcessor = faceMeshProcessor;
          _faceMeshInferencePipeline = inferencePipeline;
          _faceMeshInferenceStreamProcessor = inferenceStreamProcessor;
        });
      } else {
        _faceMeshProcessor = faceMeshProcessor;
        _faceMeshInferencePipeline = inferencePipeline;
        _faceMeshInferenceStreamProcessor = inferenceStreamProcessor;
      }
    } catch (error) {
      _errorMessage = '$error';
    } finally {
      if (mounted) {
        setState(() => _isInitializing = false);
      } else {
        _isInitializing = false;
      }
    }
  }

  FaceDetectionModel _faceDetectionModelForSelection(String value) {
    switch (value) {
      case _fullRangeDenseModel:
        return FaceDetectionModel.fullRange;
      case _fullRangeSparseModel:
        return FaceDetectionModel.fullRangeSparse;
      case _shortRangeModel:
      default:
        return FaceDetectionModel.shortRange;
    }
  }

  Future<FaceDetectorProcessor> _createFaceDetectorProcessor() {
    final model = _faceDetectionModelForSelection(_selectedModel);
    final isFullRange = model != FaceDetectionModel.shortRange;
    return FaceDetectorProcessor.create(
      model: model,
      delegate: FaceMeshDelegate.xnnpack,
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

  Future<FaceMeshProcessor> _createFaceMeshProcessor({
    required bool multi,
    required bool iris,
    bool attention = false,
  }) {
    // Multi-face tracking is managed by the pipeline with explicit per-face
    // ROIs, so the mesh processor must not keep native per-call state.
    return multi
        ? FaceMeshProcessor.createForMultiFace(
            delegate: FaceMeshDelegate.xnnpack,
            enableIris: iris,
            enableAttentionMesh: attention,
          )
        : FaceMeshProcessor.create(
            delegate: FaceMeshDelegate.xnnpack,
            enableIris: iris,
            enableAttentionMesh: attention,
          );
  }

  Future<void> _changeDetectionModel(String value) async {
    if (value == _selectedModel) {
      return;
    }
    final previousSelection = _selectedModel;
    if (mounted) {
      setState(() {
        _selectedModel = value;
        _errorMessage = null;
      });
    } else {
      _selectedModel = value;
      _errorMessage = null;
    }

    try {
      final newFaceDetectorProcessor = await _createFaceDetectorProcessor();
      _stopInferenceStream();
      _clearDetections();
      final oldProcessor = _faceDetectorProcessor;
      _faceDetectorProcessor = newFaceDetectorProcessor;
      _faceMeshInferencePipeline = FaceMeshInferencePipeline(
        detector: newFaceDetectorProcessor,
        mesh: _faceMeshProcessor,
      );
      _faceMeshInferenceStreamProcessor = FaceMeshInferenceStreamProcessor(
        _faceMeshInferencePipeline,
      );
      oldProcessor.close();
    } catch (error) {
      if (mounted) {
        setState(() {
          _selectedModel = previousSelection;
          _errorMessage = '$error';
        });
      } else {
        _selectedModel = previousSelection;
        _errorMessage = '$error';
      }
    }
  }

  Future<bool> _startFrameSource() async {
    _clearCameraFps();
    _stopInferenceStream();
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

  void _updateCameraFps(DateTime timestamp) {
    final prev = _lastCameraFrameTime;
    _lastCameraFrameTime = timestamp;
    if (prev == null) {
      return;
    }
    final elapsed = timestamp.difference(prev).inMicroseconds;
    if (elapsed <= 0) {
      return;
    }
    final fps = 1000000.0 / elapsed;
    final lastUpdate = _lastCameraFpsUpdateTime;
    if (lastUpdate != null &&
        timestamp.difference(lastUpdate) < _cameraFpsUpdateInterval) {
      return;
    }
    _lastCameraFpsUpdateTime = timestamp;
    if (mounted) {
      setState(() => _cameraFps = fps);
    } else {
      _cameraFps = fps;
    }
  }

  void _clearCameraFps() {
    _lastCameraFrameTime = null;
    _lastCameraFpsUpdateTime = null;
    _cameraFps = 0;
  }

  void _clearDetections() {
    _detectionResult = null;
    _trackedRoiOverlays = const <_TrackedRoiOverlay>[];
    _multiFaces = const <TrackedFaceMesh>[];
    _isProcessingFrame = false;
  }

  void _clearMesh() {
    _meshResult = null;
    _meshRotationCompensation = null;
  }

  void _stopInferenceStream() {
    _inferenceStreamSubscription?.cancel();
    _inferenceStreamSubscription = null;
    _inferenceStageInput.close();
    _inferenceStreamRotation = null;
    _isProcessingFrame = false;
  }

  void _ensureInferenceStageReady({
    required int rotationDegrees,
    required bool nv21,
  }) {
    if (_inferenceStreamSubscription != null &&
        _inferenceStreamRotation == rotationDegrees) {
      return;
    }
    _stopInferenceStream();
    // The input source changed (camera switch or rotation), so don't resume
    // landmark tracking on the previous feed's ROI.
    _faceMeshInferencePipeline.resetTracking();
    _inferenceStreamRotation = rotationDegrees;

    if (nv21) {
      _inferenceStageInput.nv21Controller =
          StreamController<FaceMeshNv21Image>();
      final Stream<FaceMeshNv21Image> frames =
          _inferenceStageInput.nv21Controller!.stream;
      _inferenceStreamSubscription = _isMultiFaceActive
          ? _faceMeshInferenceStreamProcessor
                .processNv21MultiFace(
                  frames,
                  maxMeshFaces: _maxMeshFaces,
                  runMeshResolver: (_) => _isMeshActive,
                  rotationDegrees: rotationDegrees,
                )
                .listen(
                  _handleMultiInferenceResult,
                  onError: _handleInferenceError,
                )
          : _faceMeshInferenceStreamProcessor
                .processNv21(
                  frames,
                  runMeshResolver: (_) => _isMeshActive,
                  rotationDegrees: rotationDegrees,
                )
                .listen(_handleInferenceResult, onError: _handleInferenceError);
    } else {
      _inferenceStageInput.bgraController = StreamController<FaceMeshImage>();
      final Stream<FaceMeshImage> frames =
          _inferenceStageInput.bgraController!.stream;
      _inferenceStreamSubscription = _isMultiFaceActive
          ? _faceMeshInferenceStreamProcessor
                .processMultiFace(
                  frames,
                  maxMeshFaces: _maxMeshFaces,
                  runMeshResolver: (_) => _isMeshActive,
                  rotationDegrees: rotationDegrees,
                )
                .listen(
                  _handleMultiInferenceResult,
                  onError: _handleInferenceError,
                )
          : _faceMeshInferenceStreamProcessor
                .process(
                  frames,
                  runMeshResolver: (_) => _isMeshActive,
                  rotationDegrees: rotationDegrees,
                )
                .listen(_handleInferenceResult, onError: _handleInferenceError);
    }
  }

  void _handleInferenceResult(FaceMeshInferenceResult result) {
    final rotationDegrees = _inferenceStreamRotation;
    _isProcessingFrame = false;
    if (rotationDegrees == null || !_isDetectionStageActive()) {
      return;
    }

    final snapshot = _DetectionSnapshot(
      result: result.detectionResult,
      rotationDegrees: rotationDegrees,
    );
    _applyDetectionStage(
      snapshot,
      hasMeshRoi: result.hasRoi,
      // On landmark-tracked frames the detector is skipped; show the tracked
      // ROI instead of a detection box.
      trackedOverlays: result.detectorRan
          ? const <_TrackedRoiOverlay>[]
          : <_TrackedRoiOverlay>[
              if (result.selectedRoi != null)
                _TrackedRoiOverlay(roi: result.selectedRoi!),
            ],
    );
    _applyMeshStage(result.meshResult);
  }

  void _handleMultiInferenceResult(FaceMeshMultiInferenceResult result) {
    _isProcessingFrame = false;
    if (_inferenceStreamRotation == null || !_isDetectionStageActive()) {
      return;
    }

    final List<TrackedFaceMesh> faces = _isMeshActive
        ? result.faces
        : const <TrackedFaceMesh>[];
    final List<_TrackedRoiOverlay> overlays = <_TrackedRoiOverlay>[
      // face.mesh.rect is the ROI this face's mesh inference actually used.
      for (final TrackedFaceMesh face in faces)
        _TrackedRoiOverlay(roi: face.mesh.rect, label: '#${face.trackId}'),
    ];

    void apply() {
      // detectionResult is null while every face slot is served by tracking.
      _detectionResult = result.detectionResult;
      _trackedRoiOverlays = overlays;
      _multiFaces = faces;
      // The single-face overlays (geometry/movement chips) stay off in
      // multi mode.
      _meshResult = null;
      _meshRotationCompensation = null;
      _movementLabel = null;
    }

    if (mounted) {
      setState(apply);
    } else {
      apply();
    }
  }

  void _handleInferenceError(Object error) {
    _isProcessingFrame = false;
    if (mounted) {
      setState(() => _errorMessage ??= '$error');
    } else {
      _errorMessage ??= '$error';
    }
  }

  void _applyMeshStage(FaceMeshResult? result) {
    final FaceMeshResult? meshResult = _isMeshActive ? result : null;
    final String? movementLabel = _resolveMovementLabel(meshResult);
    if (mounted) {
      setState(() {
        _meshResult = meshResult;
        _meshRotationCompensation = _isMeshActive && result != null ? 0 : null;
        _movementLabel = movementLabel;
      });
    } else {
      _meshResult = meshResult;
      _meshRotationCompensation = _isMeshActive && result != null ? 0 : null;
      _movementLabel = movementLabel;
    }
  }

  /// Runs the blendshapes post-processor on demand and maps the coefficients to
  /// a coarse facial movement label. Returns null when blendshapes are
  /// unavailable.
  String? _resolveMovementLabel(FaceMeshResult? result) {
    final FaceBlendshapesProcessor? processor = _blendshapesProcessor;
    // Blendshapes need the 478-landmark (iris) result; skip in base mesh mode.
    // Both the iris and attention modes provide it.
    if (result == null || processor == null || !_meshMode.has478) {
      return null;
    }
    final Map<FaceBlendshape, double>? blendshapes = processor.process(result);
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
        _stopInferenceStream();
        _clearDetections();
        _clearCameraFps();
      }

      if (mounted) {
        setState(reset);
      } else {
        reset();
      }
      _frameSource.stop();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _frameSource.onFrame = null;
    _frameSource.removeListener(_onFrameSourceChanged);
    _frameSource.dispose();
    _stopInferenceStream();
    _faceDetectorProcessor.close();
    _faceMeshProcessor.close();
    _blendshapesProcessor?.close();
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
    final mirror = _frameSource.mirrorHorizontal;
    final fpsText =
        'Cam: ${_cameraFps > 0 ? _cameraFps.toStringAsFixed(1) : '--'} fps';

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
                // Camera feed clipped to display ratio
                ClipRect(
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
                          if (isCameraAvailable && _detectionResult != null)
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: FaceDetectionPainter(
                                  result: _detectionResult!,
                                  mirrorHorizontal: mirror,
                                  showConfidence: false,
                                  showFaceBox: false,
                                  showRoiBox: true,
                                ),
                              ),
                            ),
                          if (isCameraAvailable &&
                              _trackedRoiOverlays.isNotEmpty)
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: _TrackedRoiPainter(
                                  overlays: _trackedRoiOverlays,
                                  mirrorHorizontal: mirror,
                                ),
                              ),
                            ),
                          if (isCameraAvailable && _meshResult != null)
                            RepaintBoundary(
                              child: IgnorePointer(
                                child: CustomPaint(
                                  painter: FaceMeshPainter(
                                    result: _meshResult!,
                                    irisDotRadius: 2,
                                    scaleWithFace: true,
                                    rotationDegrees:
                                        _meshRotationCompensation ?? 0,
                                    mirrorHorizontal: mirror,
                                  ),
                                ),
                              ),
                            ),
                          if (isCameraAvailable && _multiFaces.isNotEmpty)
                            RepaintBoundary(
                              child: IgnorePointer(
                                child: CustomPaint(
                                  painter: FaceMeshPainter(
                                    results: <FaceMeshResult>[
                                      for (final TrackedFaceMesh face
                                          in _multiFaces)
                                        face.mesh,
                                    ],
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
                // Chips outside ClipRect so they're always visible
                if (isCameraAvailable)
                  Positioned(top: 12, right: 12, child: _infoChip(fpsText)),
                if (_meshResult != null && _meshResult!.landmarks.length >= 468)
                  Positioned(
                    top: 12,
                    left: 12,
                    child: _infoChip(_geometryText(_meshResult!)),
                  ),
                Positioned(
                  bottom: 12,
                  left: 12,
                  child: _infoChip(_trackingChipText()),
                ),
                if (_movementLabel != null)
                  Positioned(
                    bottom: 12,
                    right: 12,
                    child: _movementChip(_movementLabel!),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  String _trackingChipText() {
    if (_isMultiFaceActive && _multiFaces.isNotEmpty) {
      return 'Tracking ${_multiFaces.length}/$_maxMeshFaces';
    }
    if (!_isMultiFaceActive && _trackedRoiOverlays.isNotEmpty) {
      return 'Tracking';
    }
    return 'Faces: ${_detectionResult?.detections.length ?? 0}';
  }

  String _geometryText(FaceMeshResult result) {
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
  String _detectMovement(Map<FaceBlendshape, double> blendshapes) {
    double v(FaceBlendshape shape) => blendshapes[shape] ?? 0;
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

  Widget _movementChip(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w700,
          fontSize: 18,
        ),
      ),
    );
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
      child: DropdownButtonFormField<String>(
        value: _selectedModel,
        isDense: true,
        borderRadius: BorderRadius.circular(12),
        style: _selectorTextStyle,
        icon: const Icon(Icons.expand_more_rounded, size: 20),
        decoration: _selectorDecoration('Detection Model'),
        items: const [
          DropdownMenuItem<String>(
            value: _shortRangeModel,
            child: Text('Short-range'),
          ),
          DropdownMenuItem<String>(
            value: _fullRangeDenseModel,
            child: Text('Full-range (dense)'),
          ),
          DropdownMenuItem<String>(
            value: _fullRangeSparseModel,
            child: Text('Full-range (sparse)'),
          ),
        ],
        onChanged: (value) {
          if (value == null || value == _selectedModel) {
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
        value: _meshMode,
        isDense: true,
        borderRadius: BorderRadius.circular(12),
        style: _selectorTextStyle,
        icon: const Icon(Icons.expand_more_rounded, size: 20),
        decoration: _selectorDecoration('Mesh Model'),
        items: [
          for (final _MeshMode mode in _MeshMode.values)
            // Attention needs custom TFLite ops the Windows runtime lacks.
            if (!(Platform.isWindows && mode.enableAttention))
              DropdownMenuItem<_MeshMode>(
                value: mode,
                child: Text(mode.label),
              ),
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
              const SizedBox(width: 8),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed:
                      (!_frameSource.canSwitch ||
                          _isChangingCamera ||
                          _isCameraBusy ||
                          !_isCameraActive ||
                          !isControllerReady)
                      ? null
                      : _switchCamera,
                  icon: const Icon(Icons.cameraswitch),
                  label: const Text('Switch'),
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
        _stopInferenceStream();
        _clearDetections();
      });
    }
    try {
      final initialized = await _startFrameSource();
      if (mounted) {
        setState(() => _isCameraActive = initialized);
      } else {
        _isCameraActive = initialized;
      }
    } finally {
      if (mounted) {
        setState(() => _isCameraBusy = false);
      } else {
        _isCameraBusy = false;
      }
    }
  }

  Future<void> _stopCamera() async {
    void reset() {
      _isCameraActive = false;
      _isDetectionActive = false;
      _isMeshActive = false;
      _clearMesh();
      _stopInferenceStream();
      _clearCameraFps();
      _clearDetections();
    }

    if (!_isCameraActive) {
      if (mounted) {
        setState(reset);
      } else {
        reset();
      }
      return;
    }
    if (mounted) {
      setState(() {
        _isCameraBusy = true;
        reset();
      });
    } else {
      _isCameraBusy = true;
      reset();
    }

    try {
      await _frameSource.stop();
    } catch (error) {
      _errorMessage ??= '$error';
    } finally {
      if (mounted) {
        setState(() => _isCameraBusy = false);
      } else {
        _isCameraBusy = false;
      }
    }
  }

  Future<void> _switchCamera() async {
    if (!_frameSource.canSwitch ||
        _isChangingCamera ||
        _isCameraBusy ||
        !_isCameraActive) {
      return;
    }

    if (mounted) {
      setState(() => _isChangingCamera = true);
    } else {
      _isChangingCamera = true;
    }

    _stopInferenceStream();
    _clearDetections();
    _clearCameraFps();

    try {
      final initialized = await _frameSource.switchSource();
      if (!initialized) {
        _errorMessage ??= _frameSource.lastError;
        if (mounted) {
          setState(() => _isCameraActive = false);
        } else {
          _isCameraActive = false;
        }
      }
    } finally {
      if (mounted) {
        setState(() => _isChangingCamera = false);
      } else {
        _isChangingCamera = false;
      }
    }
  }

  void _handleSourceFrame(DemoFrame frame) {
    if (_isProcessingFrame) {
      return;
    }
    _updateCameraFps(DateTime.now());
    if (!_frameSource.isReady || !_isCameraActive || !_isDetectionActive) {
      return;
    }
    try {
      _pushFrameToDetectionStage(frame);
    } catch (error) {
      if (mounted) {
        setState(() => _errorMessage ??= '$error');
      } else {
        _errorMessage ??= '$error';
      }
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

    final FaceMeshNv21Image? nv21Image = frame.nv21;
    if (nv21Image != null) {
      _ensureInferenceStageReady(
        rotationDegrees: rotationCompensation,
        nv21: true,
      );
      final controller = _inferenceStageInput.nv21Controller;
      if (controller == null || controller.isClosed) {
        return;
      }
      _isProcessingFrame = true;
      controller.add(nv21Image);
      return;
    }

    final FaceMeshImage? image = frame.image;
    if (image != null) {
      _ensureInferenceStageReady(
        rotationDegrees: rotationCompensation,
        nv21: false,
      );
      final controller = _inferenceStageInput.bgraController;
      if (controller == null || controller.isClosed) {
        return;
      }
      _isProcessingFrame = true;
      controller.add(image);
    }
  }

  void _applyDetectionStage(
    _DetectionSnapshot snapshot, {
    required bool hasMeshRoi,
    List<_TrackedRoiOverlay> trackedOverlays = const <_TrackedRoiOverlay>[],
  }) {
    void apply() {
      _detectionResult = snapshot.result;
      _trackedRoiOverlays = trackedOverlays;
      _multiFaces = const <TrackedFaceMesh>[];
      if (!_isMeshActive || !hasMeshRoi) {
        _meshResult = null;
        _meshRotationCompensation = null;
      }
    }

    if (mounted) {
      setState(apply);
    } else {
      apply();
    }
  }

  Future<void> _toggleDetection() async {
    if (!_frameSource.isReady || _isCameraBusy) {
      return;
    }

    if (_isDetectionActive) {
      _isProcessingFrame = false;
      if (mounted) {
        setState(() {
          _isDetectionActive = false;
          _isMeshActive = false;
          _clearMesh();
          _stopInferenceStream();
          _clearDetections();
        });
      }
      return;
    }

    try {
      await _frameSource.ensureFrames();
      if (mounted) {
        setState(() {
          _isDetectionActive = true;
          _clearDetections();
        });
      } else {
        _isDetectionActive = true;
        _clearDetections();
      }
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

    if (mounted) {
      setState(() {
        _isMeshActive = true;
        _clearMesh();
      });
    } else {
      _isMeshActive = true;
      _clearMesh();
    }
  }

  Future<void> _changeMeshMode(_MeshMode mode) async {
    if (_isCameraBusy || mode == _meshMode) return;
    final previous = _meshMode;
    try {
      await _replaceFaceMeshProcessor(
        multi: _isMultiFaceActive,
        iris: mode.enableIris,
        attention: mode.enableAttention,
      );
      if (mounted) {
        setState(() => _meshMode = mode);
      } else {
        _meshMode = mode;
      }
    } catch (error) {
      _meshMode = previous;
      if (mounted) {
        setState(() => _errorMessage = 'Mesh model change error: $error');
      }
    }
  }

  Future<void> _toggleMultiFace() async {
    if (_isCameraBusy) return;
    final nextMulti = !_isMultiFaceActive;
    try {
      await _replaceFaceMeshProcessor(
        multi: nextMulti,
        iris: _meshMode.enableIris,
        attention: _meshMode.enableAttention,
      );
      if (mounted) {
        setState(() => _isMultiFaceActive = nextMulti);
      } else {
        _isMultiFaceActive = nextMulti;
      }
    } catch (error) {
      if (mounted) {
        setState(() => _errorMessage = 'Multi-face toggle error: $error');
      }
    }
  }

  /// Swaps the mesh processor and rebuilds the pipeline; the inference stream
  /// re-subscribes with the new mode on the next camera frame.
  Future<void> _replaceFaceMeshProcessor({
    required bool multi,
    required bool iris,
    bool attention = false,
  }) async {
    final newProcessor = await _createFaceMeshProcessor(
      multi: multi,
      iris: iris,
      attention: attention,
    );
    _stopInferenceStream();
    _clearMesh();
    _clearDetections();
    final oldProcessor = _faceMeshProcessor;
    _faceMeshProcessor = newProcessor;
    _faceMeshInferencePipeline = FaceMeshInferencePipeline(
      detector: _faceDetectorProcessor,
      mesh: _faceMeshProcessor,
    );
    _faceMeshInferenceStreamProcessor = FaceMeshInferenceStreamProcessor(
      _faceMeshInferencePipeline,
    );
    oldProcessor.close();
  }
}
