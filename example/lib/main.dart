import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  MediapipeFaceMesh? _faceMesh;
  String _status = 'Initializing...';
  FaceMeshResult? _result;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final mesh = await MediapipeFaceMesh.create();
      setState(() {
        _faceMesh = mesh;
        _status = 'Engine ready. Tap the button to run a dummy inference.';
      });
      debugPrint("!!!!!!!!!!!!!!!!1111");
    } catch (error) {
      setState(() {
        _status =
            'Initialization failed (expected until a proper TFLite runtime/model is bundled): $error';
      });
      debugPrint("!!!!!!!!!!!!!!!!2222");
      debugPrint(error.toString());
    }
  }

  Future<void> _runOnce() async {
    final mesh = _faceMesh;
    if (mesh == null) {
      return;
    }
    try {
      // Create a dummy RGBA frame filled with zeros. Replace with camera pixels.
      final FaceMeshImage image = FaceMeshImage(
        pixels: Uint8List(meshInputBytes),
        width: 192,
        height: 192,
      );
      final FaceMeshResult result = mesh.process(image);
      setState(() {
        _result = result;
        _status =
            'Landmarks: ${result.landmarks.length}, score: ${result.score.toStringAsFixed(2)}';
      });
    } catch (error) {
      setState(() {
        _status = 'Inference failed: $error';
      });
    }
  }

  static int get meshInputBytes => 192 * 192 * 4;

  @override
  void dispose() {
    _faceMesh?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('MediaPipe Face Mesh'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _status,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: _faceMesh == null ? null : _runOnce,
                child: const Text('Run process()'),
              ),
              const SizedBox(height: 12),
              if (_result != null)
                Text(
                  'First landmark: '
                  'x=${_result!.landmarks.first.x.toStringAsFixed(3)}, '
                  'y=${_result!.landmarks.first.y.toStringAsFixed(3)}',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              const Spacer(),
              const Text(
                'Provide a valid mediapipe_face_mesh.tflite and TensorFlow Lite '
                'runtime libraries to get real predictions.',
              ),
            ],
          ),
        ),
      ),
    );
  }
}
