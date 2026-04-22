import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:mediapipe_face_mesh/face_mesh_stream_processor.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import '../../paint/detection_painter.dart';
import '../../utils/face_mesh_camera_image_adapter.dart';
import 'paint/face_mesh_painter.dart';

class _MlkitDetectionSnapshot {
  const _MlkitDetectionSnapshot({
    required this.faces,
    required this.rotationDegrees,
    required this.adjustedImageSize,
  });

  final List<Face> faces;
  final int rotationDegrees;
  final Size adjustedImageSize;

  Face? get primaryFace => faces.isEmpty ? null : faces.first;

  Rect? get primaryBoundingBox => primaryFace?.boundingBox;

  List<Detection> get overlayDetections {
    return faces.map((face) {
      final Rect box = face.boundingBox;
      return Detection(
        boundingBox: Rect.fromLTRB(
          (box.left / adjustedImageSize.width).clamp(0.0, 1.0),
          (box.top / adjustedImageSize.height).clamp(0.0, 1.0),
          (box.right / adjustedImageSize.width).clamp(0.0, 1.0),
          (box.bottom / adjustedImageSize.height).clamp(0.0, 1.0),
        ),
        confidence: 1.0,
        bboxLabel: face.trackingId != null ? 'Face #${face.trackingId}' : 'Face',
      );
    }).toList();
  }
}

/// Legacy ML Kit face-detection demo page kept for comparison testing.
///
/// Note: `example/pubspec.yaml` must include `google_mlkit_face_detection`
/// before this file can be used in the example app.
class MediaPipeFacePageMlkit extends StatefulWidget {
  const MediaPipeFacePageMlkit({
    super.key,
    required this.cameras,
  });

  final List<CameraDescription> cameras;

  @override
  State<MediaPipeFacePageMlkit> createState() => _MediaPipeFacePageMlkitState();
}

class _MediaPipeFacePageMlkitState extends State<MediaPipeFacePageMlkit>
    with WidgetsBindingObserver {
  static const double _boxScale = 1.2;
  static const Color _overlayColor = Colors.greenAccent;

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
  late final FaceDetector _faceDetector;
  late final FaceMeshProcessor _faceMeshProcessor;
  late final FaceMeshStreamProcessor _faceMeshStreamProcessor;
  StreamController<FaceMeshNv21Image>? _nv21StreamController;
  StreamController<FaceMeshImage>? _bgraStreamController;
  StreamSubscription<FaceMeshResult>? _meshStreamSubscription;
  _MlkitDetectionSnapshot? _latestDetectionSnapshot;
  int? _meshStreamRotation;
  bool _isMeshStreamBusy = false;
  final _MlkitInputImageConverter _inputImageConverter =
      _MlkitInputImageConverter();

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

      _faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          enableContours: true,
          enableClassification: true,
          performanceMode: FaceDetectorMode.fast,
        ),
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
      imageFormatGroup:
          Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.nv21,
    );
    _cameraController = controller;

    try {
      await controller.initialize();
      _clearCameraFps();
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
  }

  void _clearMesh() {
    _meshResult = null;
    _meshRotationCompensation = null;
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
            boxResolver: _resolveFaceMeshBoxForNv21,
            boxScale: _boxScale,
            boxMakeSquare: true,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleMeshResult, onError: _handleMeshError);
    } else if (Platform.isIOS) {
      _bgraStreamController = StreamController<FaceMeshImage>();
      _meshStreamSubscription = _faceMeshStreamProcessor
          .process(
            _bgraStreamController!.stream,
            boxResolver: _resolveFaceMeshBoxForBgra,
            boxScale: _boxScale,
            boxMakeSquare: true,
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
    _faceDetector.close();
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
        title: const Text('MLKit Det + Mediapipe Mesh'),
        titleTextStyle: const TextStyle(
          color: Colors.black,
          fontSize: 16,
        ),
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
                      _buildControlButtons(),
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
    final previewSize = isControllerReady ? controller!.value.previewSize : null;
    final nativeAspectRatio = (previewSize != null && previewSize.width != 0)
        ? previewSize.height / previewSize.width
        : 3 / 4;
    final fpsText =
        'Cam: ${_cameraFps > 0 ? _cameraFps.toStringAsFixed(1) : '--'} fps';

    return Builder(
      builder: (context) {
        final displayWidth = MediaQuery.of(context).size.width * 0.9;
        const displayAspectRatio = 3 / 4;
        final nativeHeight = displayWidth / nativeAspectRatio;

        return SizedBox(
          width: displayWidth,
          child: AspectRatio(
            aspectRatio: displayAspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
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
                                  faceBoxColor: _overlayColor,
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
                                    strokeColor: _overlayColor,
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
                if (isCameraAvailable)
                  Positioned(
                    top: 12,
                    right: 12,
                    child: _infoChip(fpsText),
                  ),
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
                  onPressed: (!_isCameraActive ||
                          _isCameraBusy ||
                          !isControllerReady ||
                          !_isDetectionActive)
                      ? null
                      : _toggleMesh,
                  style: ElevatedButton.styleFrom(
                    backgroundColor:
                        _isMeshActive ? Colors.purpleAccent : Colors.purple,
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
                  onPressed: (widget.cameras.length < 2 ||
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
    _isProcessingFrame = true;
    _handleCameraFrame(cameraImage, _cameraController!).whenComplete(() {
      _isProcessingFrame = false;
    });
  }

  Future<void> _handleCameraFrame(
    CameraImage cameraImage,
    CameraController controller,
  ) async {
    try {
      final snapshot = await _runDetectionStage(
        cameraImage: cameraImage,
        controller: controller,
      );
      if (snapshot == null || !_isDetectionStageActive()) {
        return;
      }

      _applyDetectionStage(snapshot);
      _runMeshStage(cameraImage: cameraImage, snapshot: snapshot);
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

  Future<_MlkitDetectionSnapshot?> _runDetectionStage({
    required CameraImage cameraImage,
    required CameraController controller,
  }) async {
    final rotationCompensation =
        _rotationCompensationDegrees(controller: controller);
    if (rotationCompensation == null) {
      return null;
    }

    final inputImageRotation = InputImageRotationValue.fromRawValue(
      rotationCompensation,
    );
    if (inputImageRotation == null) {
      return null;
    }

    final inputImage = _inputImageConverter.fromCameraImage(
      image: cameraImage,
      controller: controller,
      camera: _currentCamera,
      inputImageRotation: inputImageRotation,
    );
    if (inputImage == null) {
      return null;
    }

    final faces = await _faceDetector.processImage(inputImage);
    final adjustedImageSize = _adjustedImageSize(
      Size(cameraImage.width.toDouble(), cameraImage.height.toDouble()),
      inputImageRotation,
    );
    return _MlkitDetectionSnapshot(
      faces: faces,
      rotationDegrees: rotationCompensation,
      adjustedImageSize: adjustedImageSize,
    );
  }

  void _applyDetectionStage(_MlkitDetectionSnapshot snapshot) {
    _latestDetectionSnapshot = snapshot;

    if (mounted) {
      setState(() {
        _detections = snapshot.overlayDetections;
        if (!_isMeshActive || snapshot.primaryFace == null) {
          _meshResult = null;
          _meshRotationCompensation = null;
        }
      });
    } else {
      _detections = snapshot.overlayDetections;
      if (!_isMeshActive || snapshot.primaryFace == null) {
        _meshResult = null;
        _meshRotationCompensation = null;
      }
    }
  }

  void _runMeshStage({
    required CameraImage cameraImage,
    required _MlkitDetectionSnapshot snapshot,
  }) {
    if (!_isMeshActive || snapshot.primaryBoundingBox == null) {
      return;
    }
    _pushFrameToMeshStage(
      cameraImage: cameraImage,
      rotationDegrees: snapshot.rotationDegrees,
    );
  }

  Size _adjustedImageSize(Size imageSize, InputImageRotation rotation) {
    if (rotation == InputImageRotation.rotation90deg ||
        rotation == InputImageRotation.rotation270deg) {
      return Size(imageSize.height, imageSize.width);
    }
    return imageSize;
  }

  FaceMeshBox? _resolveFaceMeshBoxForNv21(FaceMeshNv21Image frame) {
    return _resolveFaceMeshBox(
      width: frame.width,
      height: frame.height,
    );
  }

  FaceMeshBox? _resolveFaceMeshBoxForBgra(FaceMeshImage frame) {
    return _resolveFaceMeshBox(
      width: frame.width,
      height: frame.height,
    );
  }

  FaceMeshBox? _resolveFaceMeshBox({
    required int width,
    required int height,
  }) {
    final snapshot = _latestDetectionSnapshot;
    final Rect? bbox = snapshot?.primaryBoundingBox;
    final int? rotationDegrees = snapshot?.rotationDegrees;
    if (bbox == null || rotationDegrees == null) {
      return null;
    }

    final inputRotation = InputImageRotationValue.fromRawValue(rotationDegrees);
    if (inputRotation == null) {
      return null;
    }

    final adjustedSize = _adjustedImageSize(
      Size(width.toDouble(), height.toDouble()),
      inputRotation,
    );
    if ((snapshot!.adjustedImageSize.width - adjustedSize.width).abs() > 0.5 ||
        (snapshot.adjustedImageSize.height - adjustedSize.height).abs() > 0.5) {
      return null;
    }

    final clamped = Rect.fromLTRB(
      bbox.left.clamp(0.0, adjustedSize.width),
      bbox.top.clamp(0.0, adjustedSize.height),
      bbox.right.clamp(0.0, adjustedSize.width),
      bbox.bottom.clamp(0.0, adjustedSize.height),
    );
    return FaceMeshBox.fromLTWH(
      left: clamped.left,
      top: clamped.top,
      width: clamped.width,
      height: clamped.height,
    );
  }

  void _pushFrameToMeshStage({
    required CameraImage cameraImage,
    required int rotationDegrees,
  }) {
    if (_latestDetectionSnapshot?.primaryBoundingBox == null ||
        _isMeshStreamBusy) {
      return;
    }
    if (!_isMeshActive ||
        !_ensureMeshStageInputReady(rotationDegrees: rotationDegrees)) {
      return;
    }

    if (Platform.isAndroid) {
      _pushNv21FrameToMeshStage(cameraImage);
    } else if (Platform.isIOS) {
      _pushBgraFrameToMeshStage(cameraImage);
    }
  }

  bool _ensureMeshStageInputReady({required int rotationDegrees}) {
    _ensureMeshStageReady(rotationDegrees: rotationDegrees);
    return _meshStreamSubscription != null;
  }

  void _pushNv21FrameToMeshStage(CameraImage cameraImage) {
    final controller = _nv21StreamController;
    final frame = FaceMeshCameraImageAdapter.toNv21(cameraImage);
    if (controller == null || controller.isClosed || frame == null) {
      return;
    }
    _isMeshStreamBusy = true;
    controller.add(frame);
  }

  void _pushBgraFrameToMeshStage(CameraImage cameraImage) {
    final controller = _bgraStreamController;
    final frame = FaceMeshCameraImageAdapter.toBgra(cameraImage);
    if (controller == null || controller.isClosed || frame == null) {
      return;
    }
    _isMeshStreamBusy = true;
    controller.add(frame);
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
        setState(() {
          _errorMessage =
              'Detection start error: ${error.description ?? error.code}';
        });
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
        setState(() => _errorMessage ??= 'Start Detect first to get a face ROI.');
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

class _MlkitInputImageConverter {
  InputImage? fromCameraImage({
    required CameraImage image,
    required CameraController controller,
    required CameraDescription camera,
    required InputImageRotation inputImageRotation,
  }) {
    final format = InputImageFormatValue.fromRawValue(image.format.raw);

    final isValidFormat = format != null &&
        ((Platform.isAndroid && format == InputImageFormat.nv21) ||
            (Platform.isIOS && format == InputImageFormat.bgra8888));

    if (!isValidFormat) {
      return null;
    }
    if (image.planes.length != 1) {
      return null;
    }

    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: inputImageRotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }
}
