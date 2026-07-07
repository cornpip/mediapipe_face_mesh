import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/face_detection_painter.dart';
import 'package:mediapipe_face_mesh/face_mesh_painter.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'utils/face_mesh_camera_image_adapter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final List<CameraDescription> cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MediaPipe Face Mesh',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: MediaPipeFacePage(cameras: cameras),
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

/// Draws the rotated ROI that landmark tracking used for mesh inference.
///
/// Shown while the detector is skipped, in place of the detection ROI box.
class _TrackedRoiPainter extends CustomPainter {
  const _TrackedRoiPainter({required this.roi, this.mirrorHorizontal = false});

  final NormalizedRect roi;
  final bool mirrorHorizontal;

  @override
  void paint(Canvas canvas, Size size) {
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

    final Paint paint = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    final Path path = Path()
      ..moveTo(corners[0].dx, corners[0].dy)
      ..lineTo(corners[1].dx, corners[1].dy)
      ..lineTo(corners[2].dx, corners[2].dy)
      ..lineTo(corners[3].dx, corners[3].dy)
      ..close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _TrackedRoiPainter oldDelegate) {
    return oldDelegate.roi != roi ||
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
  const MediaPipeFacePage({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  State<MediaPipeFacePage> createState() => _MediaPipeFacePageState();
}

class _MediaPipeFacePageState extends State<MediaPipeFacePage>
    with WidgetsBindingObserver {
  static const String _shortRangeModel = 'short_range';
  static const String _fullRangeDenseModel = 'full_range_dense';
  static const String _fullRangeSparseModel = 'full_range_sparse';
  static const Map<DeviceOrientation, int> _deviceOrientationDegrees = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  CameraController? _cameraController;
  String? _errorMessage;
  bool _isInitializing = true;
  bool _isCameraActive = false;
  bool _isCameraBusy = false;
  bool _isChangingCamera = false;
  int _currentCameraIndex = 0;
  int? _backCameraIndex;
  int? _frontCameraIndex;
  bool _isDetectionActive = false;
  bool _isMeshActive = false;
  bool _isProcessingFrame = false;
  static const Duration _cameraFpsUpdateInterval = Duration(milliseconds: 200);
  double _cameraFps = 0;
  DateTime? _lastCameraFrameTime;
  DateTime? _lastCameraFpsUpdateTime;
  FaceDetectionResult? _detectionResult;

  /// ROI reported by landmark tracking while the detector is skipped.
  NormalizedRect? _trackedRoi;
  FaceMeshResult? _meshResult;
  int? _meshRotationCompensation;
  String? _movementLabel;
  FaceBlendshapesProcessor? _blendshapesProcessor;
  late FaceDetectorProcessor _faceDetectorProcessor;
  late FaceMeshProcessor _faceMeshProcessor;
  late FaceMeshInferencePipeline _faceMeshInferencePipeline;
  late FaceMeshInferenceStreamProcessor _faceMeshInferenceStreamProcessor;
  final _inferenceStageInput = _StageInputControllers();
  StreamSubscription<FaceMeshInferenceResult>? _inferenceStreamSubscription;
  int? _inferenceStreamRotation;
  String _selectedModel = _shortRangeModel;
  bool _isIrisEnabled = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      if (widget.cameras.isEmpty) {
        throw StateError('No available cameras on this device.');
      }
      _resolveCameraIndices();

      _faceDetectorProcessor = await _createFaceDetectorProcessor();

      final faceMeshProcessor = await FaceMeshProcessor.create(
        delegate: FaceMeshDelegate.xnnpack,
        enableIris: true,
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

  void _resolveCameraIndices() {
    _backCameraIndex = _preferredCameraIndex(CameraLensDirection.back);
    _frontCameraIndex = _preferredCameraIndex(CameraLensDirection.front);
    if (_backCameraIndex != null) {
      _currentCameraIndex = _backCameraIndex!;
    } else if (_frontCameraIndex != null) {
      _currentCameraIndex = _frontCameraIndex!;
    }
  }

  int? _preferredCameraIndex(CameraLensDirection direction) {
    int? result;
    for (var i = 0; i < widget.cameras.length; i++) {
      if (widget.cameras[i].lensDirection == direction) {
        result ??= i;
      }
    }
    return result;
  }

  CameraDescription get _currentCamera => widget.cameras[_currentCameraIndex];

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
      maxResults: 1,
      // Detector ROI defaults are scaleX/scaleY = 1.5 and shiftX/shiftY = 0.0.
      // This demo keeps the default X values and only nudges Y; with landmark
      // tracking these apply to (re)acquisition frames only. Tune per
      // model/camera if the acquisition box is too loose or tight.
      roiScaleY: isFullRange ? 1.6 : 1.7,
      roiShiftY: isFullRange ? -0.1 : -0.2,
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

  Future<bool> _initializeCamera(CameraDescription description) async {
    final previousController = _cameraController;
    if (previousController != null) {
      if (previousController.value.isStreamingImages) {
        await previousController.stopImageStream();
      }
      if (mounted) {
        setState(() => _cameraController = null);
      } else {
        _cameraController = null;
      }
      await previousController.dispose();
    }

    final controller = CameraController(
      description,
      ResolutionPreset.veryHigh,
      enableAudio: false,
      imageFormatGroup: Platform.isIOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.nv21,
    );
    _cameraController = controller;

    try {
      await controller.initialize();
      _clearCameraFps();
      _stopInferenceStream();
      _clearDetections();
      await _startImageStreamIfNeeded();
      if (mounted) {
        setState(() {});
      }
      return true;
    } on CameraException catch (error) {
      await controller.dispose();
      _cameraController = null;
      _errorMessage = 'Camera error: ${error.description ?? error.code}';
      if (mounted) {
        setState(() {});
      }
      return false;
    } catch (error) {
      await controller.dispose();
      _cameraController = null;
      _errorMessage = 'Camera stream error: $error';
      if (mounted) {
        setState(() {});
      }
      return false;
    }
  }

  Future<void> _startImageStreamIfNeeded() async {
    final controller = _cameraController;
    if (controller == null || controller.value.isStreamingImages) {
      return;
    }
    await controller.startImageStream(_processCameraImage);
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
    _trackedRoi = null;
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

  void _ensureInferenceStageReady({required int rotationDegrees}) {
    if (_inferenceStreamSubscription != null &&
        _inferenceStreamRotation == rotationDegrees) {
      return;
    }
    _stopInferenceStream();
    // The input source changed (camera switch or rotation), so don't resume
    // landmark tracking on the previous feed's ROI.
    _faceMeshInferencePipeline.resetTracking();
    _inferenceStreamRotation = rotationDegrees;

    if (Platform.isAndroid) {
      _inferenceStageInput.nv21Controller =
          StreamController<FaceMeshNv21Image>();
      _inferenceStreamSubscription = _faceMeshInferenceStreamProcessor
          .processNv21(
            _inferenceStageInput.nv21Controller!.stream,
            runMeshResolver: (_) => _isMeshActive,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleInferenceResult, onError: _handleInferenceError);
    } else if (Platform.isIOS) {
      _inferenceStageInput.bgraController = StreamController<FaceMeshImage>();
      _inferenceStreamSubscription = _faceMeshInferenceStreamProcessor
          .process(
            _inferenceStageInput.bgraController!.stream,
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
      trackedRoi: result.detectorRan ? null : result.selectedRoi,
    );
    _applyMeshStage(result.meshResult);
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
    // Blendshapes need the 478-landmark (iris) result; skip when iris is off.
    if (result == null || processor == null || !_isIrisEnabled) {
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
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      void reset() {
        _cameraController = null;
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
      controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      if (_isCameraActive) {
        _initializeCamera(_currentCamera);
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _stopInferenceStream();
    _faceDetectorProcessor.close();
    _faceMeshProcessor.close();
    _blendshapesProcessor?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    final isCameraAvailable =
        _isCameraActive && controller != null && controller.value.isInitialized;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Mediapipe Det + Mediapipe Mesh'),
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
                    child: SingleChildScrollView(
                      child: Column(
                        children: [
                          _buildModelSelector(),
                          _buildControlButtons(),
                        ],
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
    final controller = _cameraController;
    final isControllerReady = controller?.value.isInitialized == true;
    final previewSize = isControllerReady
        ? controller!.value.previewSize
        : null;
    // Native sensor ratio (landscape sensor → height/width < 1).
    final nativeAspectRatio = (previewSize != null && previewSize.width != 0)
        ? previewSize.height / previewSize.width
        : 3 / 4;
    final fpsText =
        'Cam: ${_cameraFps > 0 ? _cameraFps.toStringAsFixed(1) : '--'} fps';

    return Builder(
      builder: (context) {
        final displayWidth = MediaQuery.of(context).size.width * 0.9;
        const displayAspectRatio = 3 / 4;
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
                          if (isCameraAvailable && controller != null)
                            CameraPreview(controller)
                          else
                            Container(
                              color: Colors.black12,
                              alignment: Alignment.center,
                              child: const Text(
                                'Press Start Cam',
                                style: TextStyle(color: Colors.black54),
                              ),
                            ),
                          if (isCameraAvailable &&
                              controller != null &&
                              _detectionResult != null)
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: FaceDetectionPainter(
                                  result: _detectionResult!,
                                  mirrorHorizontal:
                                      !Platform.isIOS &&
                                      controller.description.lensDirection ==
                                          CameraLensDirection.front,
                                  showConfidence: false,
                                  showFaceBox: false,
                                  showRoiBox: true,
                                ),
                              ),
                            ),
                          if (isCameraAvailable &&
                              controller != null &&
                              _trackedRoi != null)
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: _TrackedRoiPainter(
                                  roi: _trackedRoi!,
                                  mirrorHorizontal:
                                      !Platform.isIOS &&
                                      controller.description.lensDirection ==
                                          CameraLensDirection.front,
                                ),
                              ),
                            ),
                          if (isCameraAvailable &&
                              controller != null &&
                              _meshResult != null)
                            RepaintBoundary(
                              child: IgnorePointer(
                                child: CustomPaint(
                                  painter: FaceMeshPainter(
                                    result: _meshResult!,
                                    rotationDegrees:
                                        _meshRotationCompensation ?? 0,
                                    mirrorHorizontal:
                                        !Platform.isIOS &&
                                        controller.description.lensDirection ==
                                            CameraLensDirection.front,
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
                  child: _infoChip(
                    _trackedRoi != null
                        ? 'Tracking'
                        : 'Faces: ${_detectionResult?.detections.length ?? 0}',
                  ),
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

  Widget _buildModelSelector() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 6, 20, 0),
      child: DropdownButtonFormField<String>(
        value: _selectedModel,
        decoration: const InputDecoration(
          labelText: 'Detection Model',
          border: OutlineInputBorder(),
          isDense: true,
        ),
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

  Widget _buildControlButtons() {
    final controller = _cameraController;
    final isControllerReady =
        controller != null && controller.value.isInitialized;

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
                      (widget.cameras.length < 2 ||
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
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _isCameraBusy ? null : _toggleIris,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isIrisEnabled ? Colors.teal : Colors.grey,
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.remove_red_eye_outlined),
                  label: Text(_isIrisEnabled ? 'Iris On' : 'Iris Off'),
                ),
              ),
            ],
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
      final initialized = await _initializeCamera(_currentCamera);
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
    final controller = _cameraController;
    void reset() {
      _isCameraActive = false;
      _isDetectionActive = false;
      _isMeshActive = false;
      _clearMesh();
      _stopInferenceStream();
      _clearCameraFps();
      _clearDetections();
    }

    if (controller == null || !_isCameraActive) {
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
    _cameraController = null;

    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
      await controller.dispose();
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
    if (widget.cameras.length < 2 ||
        _isChangingCamera ||
        _isCameraBusy ||
        !_isCameraActive) {
      return;
    }

    final currentLens = _currentCamera.lensDirection;
    final nextIndex = currentLens == CameraLensDirection.back
        ? (_frontCameraIndex ?? _backCameraIndex)
        : (_backCameraIndex ?? _frontCameraIndex);

    if (nextIndex == null || nextIndex == _currentCameraIndex) {
      return;
    }

    if (mounted) {
      setState(() {
        _isChangingCamera = true;
        _currentCameraIndex = nextIndex;
      });
    } else {
      _isChangingCamera = true;
      _currentCameraIndex = nextIndex;
    }

    try {
      final initialized = await _initializeCamera(widget.cameras[nextIndex]);
      if (!initialized) {
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

  void _processCameraImage(CameraImage cameraImage) {
    if (_isProcessingFrame) {
      return;
    }
    _updateCameraFps(DateTime.now());
    if (_cameraController == null || !_isCameraActive || !_isDetectionActive) {
      return;
    }
    _handleCameraFrame(cameraImage, _cameraController!);
  }

  void _handleCameraFrame(
    CameraImage cameraImage,
    CameraController controller,
  ) {
    try {
      _pushFrameToDetectionStage(
        cameraImage: cameraImage,
        controller: controller,
      );
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

  void _pushFrameToDetectionStage({
    required CameraImage cameraImage,
    required CameraController controller,
  }) {
    final rotationCompensation = _rotationCompensationDegrees(
      controller: controller,
    );
    if (rotationCompensation == null) {
      return;
    }

    if (Platform.isAndroid) {
      final nv21Image = FaceMeshCameraImageAdapter.toNv21(cameraImage);
      if (nv21Image == null) {
        return;
      }
      _ensureInferenceStageReady(rotationDegrees: rotationCompensation);
      final controller = _inferenceStageInput.nv21Controller;
      if (controller == null || controller.isClosed) {
        return;
      }
      _isProcessingFrame = true;
      controller.add(nv21Image);
    } else if (Platform.isIOS) {
      final bgraImage = FaceMeshCameraImageAdapter.toBgra(cameraImage);
      if (bgraImage == null) {
        return;
      }
      _ensureInferenceStageReady(rotationDegrees: rotationCompensation);
      final controller = _inferenceStageInput.bgraController;
      if (controller == null || controller.isClosed) {
        return;
      }
      _isProcessingFrame = true;
      controller.add(bgraImage);
    }
  }

  void _applyDetectionStage(
    _DetectionSnapshot snapshot, {
    required bool hasMeshRoi,
    NormalizedRect? trackedRoi,
  }) {
    void apply() {
      _detectionResult = snapshot.result;
      _trackedRoi = trackedRoi;
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

  int? _rotationCompensationDegrees({required CameraController controller}) {
    if (Platform.isAndroid) {
      final deviceRotation =
          _deviceOrientationDegrees[controller.value.deviceOrientation];
      if (deviceRotation == null) {
        return null;
      }
      if (_currentCamera.lensDirection == CameraLensDirection.front) {
        return (_currentCamera.sensorOrientation + deviceRotation) % 360;
      }
      return (_currentCamera.sensorOrientation - deviceRotation + 360) % 360;
    }
    if (Platform.isIOS) {
      return _deviceOrientationDegrees[controller.value.deviceOrientation];
    }
    return null;
  }

  Future<void> _toggleDetection() async {
    final controller = _cameraController;
    if (controller == null ||
        !controller.value.isInitialized ||
        _isCameraBusy) {
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
      await _startImageStreamIfNeeded();
      if (mounted) {
        setState(() {
          _isDetectionActive = true;
          _clearDetections();
        });
      } else {
        _isDetectionActive = true;
        _clearDetections();
      }
    } on CameraException catch (error) {
      if (mounted) {
        setState(
          () => _errorMessage =
              'Detection start error: ${error.description ?? error.code}',
        );
      }
    }
  }

  Future<void> _toggleMesh() async {
    if (_isCameraBusy) {
      return;
    }
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
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

  Future<void> _toggleIris() async {
    if (_isCameraBusy) return;
    final nextIris = !_isIrisEnabled;
    try {
      final newProcessor = await FaceMeshProcessor.create(
        delegate: FaceMeshDelegate.xnnpack,
        enableIris: nextIris,
      );
      _stopInferenceStream();
      _clearMesh();
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
      if (mounted) {
        setState(() => _isIrisEnabled = nextIris);
      } else {
        _isIrisEnabled = nextIris;
      }
    } catch (error) {
      if (mounted) {
        setState(() => _errorMessage = 'Iris toggle error: $error');
      }
    }
  }
}
