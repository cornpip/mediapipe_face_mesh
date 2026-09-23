import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:mediapipe_face_mesh/face_detection_painter.dart';
import 'package:mediapipe_face_mesh/face_mesh_painter.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

import '../sources/frame_source.dart';

/// The camera preview with the result overlays and info chips.
///
/// Results are drawn as-is (no compensation for the Image process options),
/// so an input-side rotation or mirror is visible on screen: the mesh draws
/// where the transformed coordinates say it is. The display-only rotation
/// and flips apply to the composited preview and overlays as one layer, so
/// they cannot drift apart.
class PreviewView extends StatelessWidget {
  const PreviewView({
    super.key,
    required this.frameSource,
    required this.isCameraAvailable,
    required this.inference,
    required this.multiInference,
    required this.multiFaceLabels,
    required this.movementLabel,
    required this.geometryText,
    required this.inferenceFps,
    required this.showMesh,
    required this.isMultiFace,
    required this.maxMeshFaces,
    required this.rotationDegrees,
    required this.mirror,
    required this.flipVertical,
    required this.onSwitchCamera,
  });

  final DemoFrameSource frameSource;
  final bool isCameraAvailable;

  /// Last single-face result. Null in multi-face mode.
  final FaceMeshInferenceResult? inference;

  /// Last multi-face result. Null in single-face mode.
  final FaceMeshMultiInferenceResult? multiInference;

  /// Label per track id for the multi-face ROI boxes.
  final Map<int, String> multiFaceLabels;

  /// Label for the single-face ROI box.
  final String? movementLabel;
  final String? geometryText;
  final double inferenceFps;
  final bool showMesh;
  final bool isMultiFace;
  final int maxMeshFaces;

  /// Display-only rotation (0/90/180/270), mirror, and vertical flip.
  final int rotationDegrees;
  final bool mirror;
  final bool flipVertical;

  /// Null disables the switch button. The button is hidden when the source
  /// cannot switch.
  final VoidCallback? onSwitchCamera;

  @override
  Widget build(BuildContext context) {
    final Size screen = MediaQuery.of(context).size;
    final double displayAspectRatio = frameSource.displayAspectRatio;
    // Cap by height too so wide desktop windows keep room for controls.
    final double displayWidth = math.min(
      screen.width * 0.9,
      screen.height * 0.55 * displayAspectRatio,
    );
    // Inner SizedBox keeps the camera's native ratio so it renders correctly.
    final double nativeHeight = displayWidth / frameSource.nativeAspectRatio;
    final bool mirrorResults = frameSource.mirrorHorizontal;
    final String fpsText =
        'Infer: ${inferenceFps > 0 ? inferenceFps.toStringAsFixed(1) : '--'} fps';

    return SizedBox(
      width: displayWidth,
      child: AspectRatio(
        aspectRatio: displayAspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            ClipRect(
              child: Transform.flip(
                flipX: mirror,
                flipY: flipVertical,
                child: RotatedBox(
                  quarterTurns: rotationDegrees ~/ 90,
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: displayWidth,
                      height: nativeHeight,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          if (isCameraAvailable)
                            frameSource.buildPreview()
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
                          if (isCameraAvailable && inference != null)
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: FaceDetectionPainter.fromInference(
                                  inference!,
                                  label: movementLabel,
                                  mirrorHorizontal: mirrorResults,
                                  showConfidence: false,
                                  showFaceBox: false,
                                ),
                              ),
                            ),
                          if (isCameraAvailable && multiInference != null)
                            RepaintBoundary(
                              child: CustomPaint(
                                painter:
                                    FaceDetectionPainter.fromMultiInference(
                                      multiInference!,
                                      labelOf: (TrackedFaceMesh face) =>
                                          multiFaceLabels[face.trackId] ??
                                          '#${face.trackId}',
                                      mirrorHorizontal: mirrorResults,
                                      showConfidence: false,
                                      showFaceBox: false,
                                      // Tracked faces carry their own ROI.
                                      // Detector ROIs only matter while the
                                      // mesh is off.
                                      showRoiBox: !showMesh,
                                    ),
                              ),
                            ),
                          if (isCameraAvailable &&
                              showMesh &&
                              inference?.meshResult != null)
                            RepaintBoundary(
                              child: IgnorePointer(
                                child: CustomPaint(
                                  painter: FaceMeshPainter.fromInference(
                                    inference!,
                                    irisDotRadius: 2,
                                    scaleWithFace: true,
                                    mirrorHorizontal: mirrorResults,
                                  ),
                                ),
                              ),
                            ),
                          if (isCameraAvailable &&
                              showMesh &&
                              (multiInference?.faces.isNotEmpty ?? false))
                            RepaintBoundary(
                              child: IgnorePointer(
                                child: CustomPaint(
                                  painter: FaceMeshPainter.fromMultiInference(
                                    multiInference!,
                                    irisDotRadius: 2,
                                    scaleWithFace: true,
                                    mirrorHorizontal: mirrorResults,
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
            // Chips outside ClipRect so they're always visible.
            if (isCameraAvailable)
              Positioned(top: 12, right: 12, child: _InfoChip(fpsText)),
            if (geometryText != null)
              Positioned(top: 12, left: 12, child: _InfoChip(geometryText!)),
            if (isCameraAvailable)
              Positioned(
                bottom: 12,
                left: 12,
                child: _InfoChip(_trackingChipText()),
              ),
            if (isCameraAvailable && frameSource.canSwitch)
              Positioned(
                bottom: 12,
                right: 12,
                child: Material(
                  color: Colors.black54,
                  shape: const CircleBorder(),
                  child: IconButton(
                    icon: const Icon(Icons.cameraswitch, color: Colors.white),
                    tooltip: 'Switch camera',
                    onPressed: onSwitchCamera,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  String _trackingChipText() {
    final int trackedFaces = multiInference?.faces.length ?? 0;
    if (isMultiFace && trackedFaces > 0) {
      return 'Tracking $trackedFaces/$maxMeshFaces';
    }
    if (!isMultiFace && inference?.detectorRan == false) {
      return 'Tracking';
    }
    final FaceDetectionResult? detections =
        inference?.detectionResult ?? multiInference?.detectionResult;
    return 'Faces: ${detections?.detections.length ?? 0}';
  }
}

class _InfoChip extends StatelessWidget {
  const _InfoChip(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
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
}
