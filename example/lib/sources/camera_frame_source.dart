import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

import '../utils/face_mesh_camera_image_adapter.dart';
import 'frame_source.dart';

/// [DemoFrameSource] backed by the `camera` plugin (Android/iOS).
///
/// Android delivers NV21 frames, iOS delivers BGRA frames; both go through
/// [FaceMeshCameraImageAdapter] exactly as the demo always did.
class CameraFrameSource extends DemoFrameSource {
  CameraFrameSource(this.cameras) {
    _backCameraIndex = _preferredCameraIndex(CameraLensDirection.back);
    _frontCameraIndex = _preferredCameraIndex(CameraLensDirection.front);
    _currentCameraIndex = _backCameraIndex ?? _frontCameraIndex ?? 0;
  }

  static const Map<DeviceOrientation, int> _deviceOrientationDegrees = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  final List<CameraDescription> cameras;
  CameraController? _controller;
  int _currentCameraIndex = 0;
  int? _backCameraIndex;
  int? _frontCameraIndex;

  CameraDescription get _currentCamera => cameras[_currentCameraIndex];

  int? _preferredCameraIndex(CameraLensDirection direction) {
    for (var i = 0; i < cameras.length; i++) {
      if (cameras[i].lensDirection == direction) {
        return i;
      }
    }
    return null;
  }

  @override
  bool get isReady => _controller?.value.isInitialized == true;

  @override
  double get nativeAspectRatio {
    final Size? previewSize = _controller?.value.isInitialized == true
        ? _controller!.value.previewSize
        : null;
    // Native sensor ratio (landscape sensor → height/width < 1 upright).
    return (previewSize != null && previewSize.width != 0)
        ? previewSize.height / previewSize.width
        : 3 / 4;
  }

  @override
  double get displayAspectRatio => 3 / 4;

  @override
  int? get rotationCompensationDegrees {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return null;
    }
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

  @override
  bool get mirrorHorizontal =>
      !Platform.isIOS &&
      _currentCamera.lensDirection == CameraLensDirection.front;

  @override
  bool get canSwitch =>
      cameras.length >= 2 &&
      _backCameraIndex != null &&
      _frontCameraIndex != null;

  @override
  bool get supportsLifecyclePause => true;

  @override
  Future<bool> start() async {
    if (cameras.isEmpty) {
      lastError = 'No available cameras on this device.';
      return false;
    }
    return _initializeController(_currentCamera);
  }

  Future<bool> _initializeController(CameraDescription description) async {
    final previousController = _controller;
    if (previousController != null) {
      _controller = null;
      notifyListeners();
      if (previousController.value.isStreamingImages) {
        await previousController.stopImageStream();
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
    _controller = controller;

    try {
      await controller.initialize();
      await ensureFrames();
      notifyListeners();
      return true;
    } on CameraException catch (error) {
      await controller.dispose();
      _controller = null;
      lastError = 'Camera error: ${error.description ?? error.code}';
      notifyListeners();
      return false;
    } catch (error) {
      await controller.dispose();
      _controller = null;
      lastError = 'Camera stream error: $error';
      notifyListeners();
      return false;
    }
  }

  @override
  Future<void> ensureFrames() async {
    final controller = _controller;
    if (controller == null || controller.value.isStreamingImages) {
      return;
    }
    await controller.startImageStream(_handleCameraImage);
  }

  void _handleCameraImage(CameraImage cameraImage) {
    final callback = onFrame;
    if (callback == null) {
      return;
    }
    if (Platform.isAndroid) {
      final nv21Image = FaceMeshCameraImageAdapter.toNv21(cameraImage);
      if (nv21Image != null) {
        callback(DemoFrame.nv21(nv21Image));
      }
    } else if (Platform.isIOS) {
      final bgraImage = FaceMeshCameraImageAdapter.toBgra(cameraImage);
      if (bgraImage != null) {
        callback(DemoFrame.image(bgraImage));
      }
    }
  }

  @override
  Future<void> stop() async {
    final controller = _controller;
    _controller = null;
    notifyListeners();
    if (controller == null) {
      return;
    }
    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
      await controller.dispose();
    } catch (error) {
      lastError ??= '$error';
    }
  }

  @override
  Future<bool> switchSource() async {
    final currentLens = _currentCamera.lensDirection;
    final nextIndex = currentLens == CameraLensDirection.back
        ? (_frontCameraIndex ?? _backCameraIndex)
        : (_backCameraIndex ?? _frontCameraIndex);
    if (nextIndex == null || nextIndex == _currentCameraIndex) {
      return true;
    }
    _currentCameraIndex = nextIndex;
    return _initializeController(cameras[nextIndex]);
  }

  @override
  Widget buildPreview() {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const SizedBox.shrink();
    }
    return CameraPreview(controller);
  }

  @override
  void dispose() {
    _controller?.dispose();
    _controller = null;
    super.dispose();
  }
}
