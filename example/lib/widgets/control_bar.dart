import 'package:flutter/material.dart';

/// Start Cam and Start Detect, pinned below the preview.
///
/// Starting the camera also starts detection. Stop Detect keeps the camera
/// running and stops handing frames to the pipeline. A null callback
/// disables its button.
class ControlBar extends StatelessWidget {
  const ControlBar({
    super.key,
    required this.isCameraActive,
    required this.isDetectionActive,
    required this.onToggleCamera,
    required this.onToggleDetection,
  });

  final bool isCameraActive;
  final bool isDetectionActive;
  final VoidCallback? onToggleCamera;
  final VoidCallback? onToggleDetection;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 0),
      child: Row(
        children: [
          Expanded(
            child: ElevatedButton.icon(
              onPressed: onToggleCamera,
              style: ElevatedButton.styleFrom(
                backgroundColor: isCameraActive
                    ? Colors.redAccent
                    : Colors.greenAccent,
                foregroundColor: Colors.black,
              ),
              icon: Icon(
                isCameraActive ? Icons.stop : Icons.videocam,
                color: Colors.black,
              ),
              label: Text(isCameraActive ? 'Stop Cam' : 'Start Cam'),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: ElevatedButton.icon(
              onPressed: onToggleDetection,
              style: ElevatedButton.styleFrom(
                backgroundColor: isDetectionActive
                    ? Colors.orangeAccent
                    : Colors.blueAccent,
                foregroundColor: Colors.black,
              ),
              icon: Icon(
                isDetectionActive ? Icons.pause : Icons.play_arrow,
                color: Colors.black,
              ),
              label: Text(isDetectionActive ? 'Stop Detect' : 'Start Detect'),
            ),
          ),
        ],
      ),
    );
  }
}
