import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mediapipe_face_mesh/face_mesh_stream_processor.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import '../../paint/detection_painter.dart';
import '../../utils/face_mesh_camera_image_adapter.dart';
import 'paint/face_mesh_painter.dart';

class _DetectionSnapshot {
  const _DetectionSnapshot({
    required this.result,
    required this.rotationDegrees,
  });

  final FaceDetectionResult result;
  final int rotationDegrees;

  FaceDetection? get primaryDetection => result.primaryDetection;

  Size get imageSize =>
      Size(result.imageWidth.toDouble(), result.imageHeight.toDouble());

  NormalizedRect? get primaryRoi => primaryDetection?.expandedFaceRect;

  List<Detection> get overlayDetections {
    return result.detections.map((detection) {
      return Detection(
        boundingBox: Rect.fromLTRB(
          detection.left.clamp(0.0, 1.0),
          detection.top.clamp(0.0, 1.0),
          detection.right.clamp(0.0, 1.0),
          detection.bottom.clamp(0.0, 1.0),
        ),
        confidence: detection.score,
        bboxLabel: 'Face',
        roiLabel: 'ROI',
        rotatedRect: detection.expandedFaceRect,
      );
    }).toList();
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
  List<Detection> _detections = const [];
  FaceMeshResult? _meshResult;
  int? _meshRotationCompensation;
  late final FaceDetectorProcessor _faceDetectorProcessor;
  late final FaceDetectorStreamProcessor _faceDetectorStreamProcessor;
  late final FaceMeshProcessor _faceMeshProcessor;
  late final FaceMeshStreamProcessor _faceMeshStreamProcessor;
  StreamController<FaceMeshNv21Image>? _detectorNv21StreamController;
  StreamController<FaceMeshImage>? _detectorBgraStreamController;
  StreamController<FaceMeshNv21Image>? _nv21StreamController;
  StreamController<FaceMeshImage>? _bgraStreamController;
  StreamSubscription<FaceDetectionResult>? _detectorStreamSubscription;
  StreamSubscription<FaceMeshResult>? _meshStreamSubscription;
  _DetectionSnapshot? _latestDetectionSnapshot;
  Object? _pendingDetectorFrame;
  int? _pendingDetectorRotation;
  int? _detectorStreamRotation;
  int? _meshStreamRotation;
  bool _isMeshStreamBusy = false;
  String _selectedModel = _shortRangeModel;

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

      _faceDetectorProcessor = await FaceDetectorProcessor.create(
        delegate: FaceMeshDelegate.xnnpack,
        maxResults: 1,
        roiScaleY: 1.7,
        roiShiftY: -0.2,
      );
      _faceDetectorStreamProcessor = FaceDetectorStreamProcessor(
        _faceDetectorProcessor,
      );

      final faceMeshProcessor = await FaceMeshProcessor.create(
        delegate: FaceMeshDelegate.xnnpack,
      );
      if (mounted) {
        setState(() {
          _faceMeshProcessor = faceMeshProcessor;
          _faceMeshStreamProcessor = FaceMeshStreamProcessor(faceMeshProcessor);
        });
      } else {
        _faceMeshProcessor = faceMeshProcessor;
        _faceMeshStreamProcessor = FaceMeshStreamProcessor(faceMeshProcessor);
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
      _stopDetectorStream();
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
    _detections = const [];
    _isProcessingFrame = false;
    _latestDetectionSnapshot = null;
    _pendingDetectorFrame = null;
    _pendingDetectorRotation = null;
  }

  void _clearMesh() {
    _meshResult = null;
    _meshRotationCompensation = null;
  }

  void _stopDetectorStream() {
    _detectorStreamSubscription?.cancel();
    _detectorStreamSubscription = null;
    _detectorNv21StreamController?.close();
    _detectorBgraStreamController?.close();
    _detectorNv21StreamController = null;
    _detectorBgraStreamController = null;
    _pendingDetectorFrame = null;
    _pendingDetectorRotation = null;
    _detectorStreamRotation = null;
    _isProcessingFrame = false;
  }

  void _ensureDetectorStageReady({required int rotationDegrees}) {
    if (_detectorStreamSubscription != null &&
        _detectorStreamRotation == rotationDegrees) {
      return;
    }
    _stopDetectorStream();
    _detectorStreamRotation = rotationDegrees;

    if (Platform.isAndroid) {
      _detectorNv21StreamController = StreamController<FaceMeshNv21Image>();
      _detectorStreamSubscription = _faceDetectorStreamProcessor
          .processNv21(
            _detectorNv21StreamController!.stream,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleDetectorResult, onError: _handleDetectorError);
    } else if (Platform.isIOS) {
      _detectorBgraStreamController = StreamController<FaceMeshImage>();
      _detectorStreamSubscription = _faceDetectorStreamProcessor
          .process(
            _detectorBgraStreamController!.stream,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleDetectorResult, onError: _handleDetectorError);
    }
  }

  void _handleDetectorResult(FaceDetectionResult result) {
    final frame = _pendingDetectorFrame;
    final rotationDegrees = _pendingDetectorRotation;
    _pendingDetectorFrame = null;
    _pendingDetectorRotation = null;
    _isProcessingFrame = false;
    if (frame == null ||
        rotationDegrees == null ||
        !_isDetectionStageActive()) {
      return;
    }

    final snapshot = _DetectionSnapshot(
      result: result,
      rotationDegrees: rotationDegrees,
    );
    _applyDetectionStage(snapshot);
    _runMeshStage(frame: frame, snapshot: snapshot);
  }

  void _handleDetectorError(Object error) {
    _pendingDetectorFrame = null;
    _pendingDetectorRotation = null;
    _isProcessingFrame = false;
    if (mounted) {
      setState(() => _errorMessage ??= '$error');
    } else {
      _errorMessage ??= '$error';
    }
  }

  void _stopMeshStream() {
    _meshStreamSubscription?.cancel();
    _meshStreamSubscription = null;
    _nv21StreamController?.close();
    _bgraStreamController?.close();
    _nv21StreamController = null;
    _bgraStreamController = null;
    _meshStreamRotation = null;
    _isMeshStreamBusy = false;
  }

  void _ensureMeshStageReady({required int rotationDegrees}) {
    if (_meshStreamSubscription != null &&
        _meshStreamRotation == rotationDegrees) {
      return;
    }
    _stopMeshStream();
    _meshStreamRotation = rotationDegrees;

    if (Platform.isAndroid) {
      _nv21StreamController = StreamController<FaceMeshNv21Image>();
      _meshStreamSubscription = _faceMeshStreamProcessor
          .processNv21(
            _nv21StreamController!.stream,
            roiResolver: _resolveFaceMeshRoi,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleMeshResult, onError: _handleMeshError);
    } else if (Platform.isIOS) {
      _bgraStreamController = StreamController<FaceMeshImage>();
      _meshStreamSubscription = _faceMeshStreamProcessor
          .process(
            _bgraStreamController!.stream,
            roiResolver: _resolveFaceMeshRoi,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleMeshResult, onError: _handleMeshError);
    }
  }

  void _handleMeshResult(FaceMeshResult result) {
    _isMeshStreamBusy = false;
    if (!_isMeshActive) {
      return;
    }
    if (mounted) {
      setState(() {
        _meshResult = result;
        _meshRotationCompensation = 0;
      });
    } else {
      _meshResult = result;
      _meshRotationCompensation = 0;
    }
  }

  void _handleMeshError(Object error) {
    _isMeshStreamBusy = false;
    if (mounted) {
      setState(() => _errorMessage ??= '$error');
    } else {
      _errorMessage ??= '$error';
    }
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
        _stopDetectorStream();
        _stopMeshStream();
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
    _stopDetectorStream();
    _faceDetectorProcessor.close();
    _stopMeshStream();
    _faceMeshProcessor.close();
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
                          if (isCameraAvailable && controller != null)
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: DetectionPainter(
                                  detections: _detections,
                                  lensDirection:
                                      controller.description.lensDirection,
                                  showConfidence: false,
                                  showFaceBox: false,
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
                                    rotationCompensation:
                                        _meshRotationCompensation ?? 0,
                                    lensDirection:
                                        controller.description.lensDirection,
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
                Positioned(
                  bottom: 12,
                  left: 12,
                  child: _infoChip('Faces: ${_detections.length}'),
                ),
              ],
            ),
          ),
        );
      },
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
        initialValue: _selectedModel,
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
            enabled: false,
            child: Text(
              'Full-range (dense) - Planned',
              style: TextStyle(color: Colors.black38),
            ),
          ),
          DropdownMenuItem<String>(
            value: _fullRangeSparseModel,
            enabled: false,
            child: Text(
              'Full-range (sparse) - Planned',
              style: TextStyle(color: Colors.black38),
            ),
          ),
        ],
        onChanged: (value) {
          if (value == null || value == _selectedModel) {
            return;
          }
          setState(() => _selectedModel = value);
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
        _stopDetectorStream();
        _stopMeshStream();
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
      _stopDetectorStream();
      _stopMeshStream();
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
      _ensureDetectorStageReady(rotationDegrees: rotationCompensation);
      final controller = _detectorNv21StreamController;
      if (controller == null || controller.isClosed) {
        return;
      }
      _pendingDetectorFrame = nv21Image;
      _pendingDetectorRotation = rotationCompensation;
      _isProcessingFrame = true;
      controller.add(nv21Image);
    } else if (Platform.isIOS) {
      final bgraImage = FaceMeshCameraImageAdapter.toBgra(cameraImage);
      if (bgraImage == null) {
        return;
      }
      _ensureDetectorStageReady(rotationDegrees: rotationCompensation);
      final controller = _detectorBgraStreamController;
      if (controller == null || controller.isClosed) {
        return;
      }
      _pendingDetectorFrame = bgraImage;
      _pendingDetectorRotation = rotationCompensation;
      _isProcessingFrame = true;
      controller.add(bgraImage);
    }
  }

  void _applyDetectionStage(_DetectionSnapshot snapshot) {
    _latestDetectionSnapshot = snapshot;

    if (mounted) {
      setState(() {
        _detections = snapshot.overlayDetections;
        if (!_isMeshActive || snapshot.primaryDetection == null) {
          _meshResult = null;
          _meshRotationCompensation = null;
        }
      });
    } else {
      _detections = snapshot.overlayDetections;
      if (!_isMeshActive || snapshot.primaryDetection == null) {
        _meshResult = null;
        _meshRotationCompensation = null;
      }
    }
  }

  bool _runMeshStage({
    required Object frame,
    required _DetectionSnapshot snapshot,
  }) {
    if (!_isMeshActive || snapshot.primaryRoi == null) {
      return false;
    }
    return _pushFrameToMeshStage(
      frame: frame,
      rotationDegrees: snapshot.rotationDegrees,
    );
  }

  NormalizedRect? _resolveFaceMeshRoi(dynamic frame) {
    final int width;
    final int height;
    if (frame is FaceMeshNv21Image) {
      width = frame.width;
      height = frame.height;
    } else if (frame is FaceMeshImage) {
      width = frame.width;
      height = frame.height;
    } else {
      return null;
    }

    final snapshot = _latestDetectionSnapshot;
    if (snapshot == null) {
      return null;
    }
    final rotationDegrees = snapshot.rotationDegrees;
    final bool swapAxes = rotationDegrees == 90 || rotationDegrees == 270;
    final double logicalWidth = swapAxes ? height.toDouble() : width.toDouble();
    final double logicalHeight = swapAxes
        ? width.toDouble()
        : height.toDouble();
    if ((snapshot.imageSize.width - logicalWidth).abs() > 0.5 ||
        (snapshot.imageSize.height - logicalHeight).abs() > 0.5) {
      return null;
    }
    return snapshot.primaryRoi;
  }

  bool _pushFrameToMeshStage({
    required Object frame,
    required int rotationDegrees,
  }) {
    if (_latestDetectionSnapshot == null || _isMeshStreamBusy) {
      return false;
    }
    if (!_isMeshActive ||
        !_ensureMeshStageInputReady(rotationDegrees: rotationDegrees)) {
      return false;
    }

    if (frame is FaceMeshNv21Image) {
      return _pushNv21FrameToMeshStage(frame);
    } else if (frame is FaceMeshImage) {
      return _pushBgraFrameToMeshStage(frame);
    }
    return false;
  }

  bool _ensureMeshStageInputReady({required int rotationDegrees}) {
    _ensureMeshStageReady(rotationDegrees: rotationDegrees);
    return _meshStreamSubscription != null;
  }

  bool _pushNv21FrameToMeshStage(FaceMeshNv21Image frame) {
    final controller = _nv21StreamController;
    if (controller == null || controller.isClosed) {
      return false;
    }
    _isMeshStreamBusy = true;
    controller.add(frame);
    return true;
  }

  bool _pushBgraFrameToMeshStage(FaceMeshImage frame) {
    final controller = _bgraStreamController;
    if (controller == null || controller.isClosed) {
      return false;
    }
    _isMeshStreamBusy = true;
    controller.add(frame);
    return true;
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
          _stopDetectorStream();
          _stopMeshStream();
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
          _stopMeshStream();
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

    final rotation = _latestDetectionSnapshot?.rotationDegrees;
    if (rotation != null) {
      _ensureMeshStageReady(rotationDegrees: rotation);
    }
  }
}
