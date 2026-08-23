import 'dart:async';

import 'package:flutter/widgets.dart';
import 'package:flutter_ffi_uvc/flutter_ffi_uvc.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import 'frame_source.dart';

/// [DemoFrameSource] backed by a USB (UVC) camera via `flutter_ffi_uvc`
/// (Windows desktop).
///
/// Device open and preview start are decoupled: the selected device is opened
/// eagerly so its camera modes are listed (highest resolution first) before
/// the preview runs. The preview renders through the plugin's native texture;
/// inference frames are RGBA copies of the shared preview buffer, polled by
/// sequence number, which feed [FaceMeshImage] without any pixel format
/// conversion.
class UvcFrameSource extends DemoFrameSource {
  UvcFrameSource() {
    _deviceEventSub = _camera.deviceEvents.listen(_onDeviceEvent);
    _initDevices();
  }

  final UvcCamera _camera = uvcCamera;

  List<UvcUsbDevice> _devices = const [];
  int _deviceIndex = 0;
  bool _deviceOpen = false;

  /// All modes of the open device, highest resolution first.
  List<UvcCameraMode> _modes = const [];

  /// Mode shown as selected in the dropdown (start target).
  UvcCameraMode? _selectedMode;

  /// Mode the running preview actually uses.
  UvcCameraMode? _runningMode;

  /// Frame format filter for the mode dropdown; null shows every format.
  String? _formatFilter;

  bool _previewRunning = false;
  int? _textureId;
  Timer? _pollTimer;
  int _lastDeliveredSequence = -1;
  StreamSubscription<UvcDeviceEvent>? _deviceEventSub;

  Future<void> _initDevices() async {
    try {
      _devices = await _camera.listUsbDevices();
      if (_deviceIndex >= _devices.length) {
        _deviceIndex = 0;
      }
      if (_devices.isNotEmpty && !_deviceOpen) {
        await _openSelectedDevice();
      }
      notifyListeners();
    } catch (error) {
      lastError = 'USB camera discovery failed: $error';
      notifyListeners();
    }
  }

  void _onDeviceEvent(UvcDeviceEvent _) {
    if (_deviceOpen) {
      // A detach of the open device surfaces through stream errors and a
      // user-driven stop/start; only refresh while idle.
      return;
    }
    _initDevices();
  }

  /// Opens the selected device (no preview yet) and loads its mode list so
  /// the mode dropdown is populated before Start Cam.
  Future<void> _openSelectedDevice() async {
    await _closeDevice();
    final UvcUsbDevice device = _devices[_deviceIndex];
    await _camera.openUsbDevice(device.deviceId);
    _deviceOpen = true;
    _modes = _sortModes(_camera.supportedModes());
    _selectedMode = _modes.isNotEmpty ? _modes.first : null;
    _runningMode = null;
    _formatFilter = null;
    notifyListeners();
  }

  /// Highest resolution first, then highest frame rate.
  static List<UvcCameraMode> _sortModes(List<UvcCameraMode> modes) {
    final List<UvcCameraMode> sorted = List<UvcCameraMode>.from(modes);
    sorted.sort((UvcCameraMode a, UvcCameraMode b) {
      final int byArea = (b.width * b.height) - (a.width * a.height);
      if (byArea != 0) {
        return byArea;
      }
      return b.fps - a.fps;
    });
    return sorted;
  }

  List<String> get _formatNames =>
      _modes.map((UvcCameraMode mode) => mode.formatName).toSet().toList();

  List<UvcCameraMode> get _filteredModes => _formatFilter == null
      ? _modes
      : _modes
            .where((UvcCameraMode mode) => mode.formatName == _formatFilter)
            .toList();

  @override
  bool get isReady =>
      _previewRunning && _textureId != null && _runningMode != null;

  @override
  double get nativeAspectRatio {
    final UvcCameraMode? mode = _runningMode ?? _selectedMode;
    return (mode != null && mode.height != 0)
        ? mode.width / mode.height
        : 4 / 3;
  }

  @override
  double get displayAspectRatio => nativeAspectRatio;

  @override
  int? get rotationCompensationDegrees => isReady ? 0 : null;

  @override
  bool get mirrorHorizontal => false;

  @override
  bool get canSwitch => _devices.length >= 2;

  @override
  String get activeSourceLabel =>
      _devices.isEmpty ? '' : _devices[_deviceIndex].displayName;

  @override
  bool get supportsLifecyclePause => false;

  @override
  List<FrameSourceSelector> get selectors {
    final List<UvcCameraMode> filteredModes = _filteredModes;
    final UvcCameraMode? selectedMode = _selectedMode;
    return [
      if (_devices.isNotEmpty)
        FrameSourceSelector(
          label: 'USB Device',
          options: [for (final device in _devices) device.displayName],
          selectedIndex: _deviceIndex,
          onSelect: _selectDevice,
        ),
      if (filteredModes.isNotEmpty)
        FrameSourceSelector(
          label: 'Camera Mode',
          options: [for (final mode in filteredModes) mode.label],
          selectedIndex: selectedMode == null
              ? -1
              : filteredModes.indexOf(selectedMode),
          onSelect: _selectMode,
        ),
    ];
  }

  @override
  List<FrameSourceTagFilter> get tagFilters {
    final List<String> formats = _formatNames;
    if (formats.length < 2) {
      return const [];
    }
    return [
      FrameSourceTagFilter(
        options: ['All', ...formats],
        selectedIndex: _formatFilter == null
            ? 0
            : formats.indexOf(_formatFilter!) + 1,
        onSelect: (index) {
          _formatFilter = index == 0 ? null : formats[index - 1];
          notifyListeners();
        },
      ),
    ];
  }

  @override
  Future<bool> start() async {
    try {
      if (!_deviceOpen) {
        await _initDevices();
      }
      if (!_deviceOpen || _devices.isEmpty) {
        lastError ??= 'No USB (UVC) camera connected.';
        notifyListeners();
        return false;
      }
      _textureId ??= await _camera.createPreviewTexture();

      final UvcCameraMode? preferred = _selectedMode;
      UvcCameraMode? startedMode;
      if (preferred != null) {
        final UvcPreviewStartResult result = await _camera.startPreview(
          preferred,
          policy: UvcPreviewPolicy.sequenceOnly,
        );
        if (result.success) {
          startedMode = preferred;
        }
      }
      if (startedMode == null) {
        // Selected mode failed to verify — fall back to the library's
        // MJPEG-first reliability probe.
        final UvcAutoPreviewResult autoResult = await _camera
            .startPreviewAuto();
        startedMode = autoResult.mode;
      }
      if (startedMode == null) {
        lastError =
            'No working preview mode on '
            '${_devices[_deviceIndex].displayName}.';
        await _teardownPreview();
        notifyListeners();
        return false;
      }
      await _runPreview(startedMode);
      notifyListeners();
      return true;
    } catch (error) {
      lastError = 'Failed to start USB preview: $error';
      await _teardownPreview();
      notifyListeners();
      return false;
    }
  }

  Future<void> _runPreview(UvcCameraMode mode) async {
    final int? textureId = _textureId;
    if (textureId != null) {
      await _camera.attachPreviewTexture(
        textureId,
        width: mode.width,
        height: mode.height,
      );
    }
    _runningMode = mode;
    _selectedMode = mode;
    _previewRunning = true;
    _lastDeliveredSequence = -1;
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(
      const Duration(milliseconds: 33),
      (_) => _deliverLatestFrame(),
    );
  }

  void _deliverLatestFrame() {
    final callback = onFrame;
    if (callback == null || !isReady) {
      return;
    }
    final int sequence = _camera.latestFrameSequence();
    if (sequence == _lastDeliveredSequence) {
      return;
    }
    final UvcPreviewFrame? frame = _camera.copyLatestFrame();
    if (frame == null) {
      return;
    }
    _lastDeliveredSequence = sequence;
    callback(
      DemoFrame.image(
        FaceMeshImage(
          pixels: frame.rgbaBytes,
          width: frame.width,
          height: frame.height,
        ),
      ),
    );
  }

  Future<void> _selectDevice(int index) async {
    if (index < 0 || index >= _devices.length || index == _deviceIndex) {
      return;
    }
    final bool wasPreviewing = _previewRunning;
    _deviceIndex = index;
    try {
      await _teardownPreview();
      await _openSelectedDevice();
      if (wasPreviewing) {
        await start();
      }
    } catch (error) {
      lastError = 'Failed to open USB device: $error';
    }
    notifyListeners();
  }

  Future<void> _selectMode(int filteredIndex) async {
    final List<UvcCameraMode> filteredModes = _filteredModes;
    if (filteredIndex < 0 || filteredIndex >= filteredModes.length) {
      return;
    }
    final UvcCameraMode mode = filteredModes[filteredIndex];
    if (mode == _selectedMode && mode == _runningMode) {
      return;
    }
    _selectedMode = mode;
    if (!_previewRunning) {
      // Preview not started yet — the selection is just the start target.
      notifyListeners();
      return;
    }

    // Live mode switch: stop the running preview first, then verify the new
    // mode with the lenient sequence-only policy (mirrors the flutter_ffi_uvc
    // example). Restarting while streaming makes verification fail and the
    // old auto-recovery snap back to the previous mode.
    final UvcCameraMode? previous = _runningMode;
    _pollTimer?.cancel();
    _pollTimer = null;
    _camera.stopPreview();
    final UvcPreviewStartResult result = await _camera.startPreview(
      mode,
      policy: UvcPreviewPolicy.sequenceOnly,
    );
    if (result.success) {
      await _runPreview(mode);
    } else {
      lastError = 'Mode ${mode.label} failed to start.';
      // Recover the mode that was running before.
      if (previous != null) {
        final UvcPreviewStartResult recovery = await _camera.startPreview(
          previous,
          policy: UvcPreviewPolicy.sequenceOnly,
        );
        if (recovery.success) {
          await _runPreview(previous);
        } else {
          _previewRunning = false;
          _runningMode = null;
        }
      } else {
        _previewRunning = false;
        _runningMode = null;
      }
    }
    notifyListeners();
  }

  @override
  Future<bool> switchSource() async {
    if (_devices.length < 2) {
      return true;
    }
    await _selectDevice((_deviceIndex + 1) % _devices.length);
    return _previewRunning || _deviceOpen;
  }

  @override
  Future<void> ensureFrames() async {
    if (isReady && _pollTimer == null && _runningMode != null) {
      await _runPreview(_runningMode!);
    }
  }

  /// Stops the preview but keeps the device open so the mode list stays
  /// visible; the device is fully closed in [dispose] or on device switch.
  @override
  Future<void> stop() async {
    await _teardownPreview();
    notifyListeners();
  }

  Future<void> _teardownPreview() async {
    _pollTimer?.cancel();
    _pollTimer = null;
    final int? textureId = _textureId;
    _textureId = null;
    _previewRunning = false;
    _runningMode = null;
    if (_deviceOpen) {
      _camera.stopPreview();
    }
    if (textureId != null) {
      await _camera.disposePreviewTexture(textureId);
    }
  }

  Future<void> _closeDevice() async {
    await _teardownPreview();
    if (_deviceOpen) {
      _deviceOpen = false;
      await _camera.closeUsbDevice();
    }
    _modes = const [];
    _selectedMode = null;
    _formatFilter = null;
  }

  @override
  Widget buildPreview() {
    final int? textureId = _textureId;
    if (textureId == null) {
      return const SizedBox.shrink();
    }
    return Texture(textureId: textureId);
  }

  @override
  void dispose() {
    _deviceEventSub?.cancel();
    _closeDevice();
    super.dispose();
  }
}
