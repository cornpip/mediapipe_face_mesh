import 'package:flutter/widgets.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// One frame delivered by a [DemoFrameSource] — either an NV21 buffer
/// (Android camera) or an RGBA/BGRA buffer (iOS camera, UVC webcam).
class DemoFrame {
  const DemoFrame.nv21(FaceMeshNv21Image this.nv21) : image = null;
  const DemoFrame.image(FaceMeshImage this.image) : nv21 = null;

  final FaceMeshNv21Image? nv21;
  final FaceMeshImage? image;
}

/// A generic dropdown the frame source wants the demo page to render
/// (e.g. UVC device or camera mode selection).
class FrameSourceSelector {
  const FrameSourceSelector({
    required this.label,
    required this.options,
    required this.selectedIndex,
    required this.onSelect,
  });

  final String label;
  final List<String> options;

  /// Index into [options]; -1 renders the dropdown without a selection.
  final int selectedIndex;
  final Future<void> Function(int index) onSelect;
}

/// A generic single-choice chip row the frame source wants the page to render
/// (e.g. UVC frame format filter for the mode list).
class FrameSourceTagFilter {
  const FrameSourceTagFilter({
    required this.options,
    required this.selectedIndex,
    required this.onSelect,
  });

  final List<String> options;
  final int selectedIndex;
  final void Function(int index) onSelect;
}

/// Abstraction over where demo frames come from, so the demo page's inference
/// and overlay UI is identical across the mobile `camera` plugin and the
/// Windows USB (UVC) camera.
///
/// Implementations call [notifyListeners] whenever preview state or the
/// selector lists change, and push frames through [onFrame] while started.
abstract class DemoFrameSource extends ChangeNotifier {
  /// Set by the page; receives frames while the source is started.
  void Function(DemoFrame frame)? onFrame;

  /// Last start/stream failure, for the page's error banner.
  String? lastError;

  /// Whether the preview is initialized and delivering frames.
  bool get isReady;

  /// Upright preview width / height.
  double get nativeAspectRatio;

  /// Aspect ratio of the preview box in the page layout.
  double get displayAspectRatio;

  /// Rotation to apply to frames before inference; null while unknown.
  int? get rotationCompensationDegrees;

  /// Whether overlays must be mirrored to match the preview.
  bool get mirrorHorizontal;

  /// Whether [switchSource] can do anything (second lens / second device).
  bool get canSwitch;

  /// Short label of the active source shown next to the switch control
  /// ('front'/'back' for the mobile camera, the device name for UVC).
  String get activeSourceLabel => '';

  /// Whether the app lifecycle should stop/restart this source (mobile
  /// camera). Desktop windows lose focus constantly, so UVC returns false.
  bool get supportsLifecyclePause;

  /// Extra dropdowns the page should render for this source.
  List<FrameSourceSelector> get selectors => const [];

  /// Extra chip-row filters the page should render above [selectors].
  List<FrameSourceTagFilter> get tagFilters => const [];

  /// Starts the preview and frame delivery. Returns false and sets
  /// [lastError] on failure.
  Future<bool> start();

  Future<void> stop();

  /// Switches to the other lens / next device while running.
  Future<bool> switchSource();

  /// Re-arms frame delivery if it was stopped (no-op when already flowing).
  Future<void> ensureFrames();

  /// The preview widget for the current state.
  Widget buildPreview();
}
