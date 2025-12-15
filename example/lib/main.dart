import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _FaceMeshPainter extends CustomPainter {
  _FaceMeshPainter({
    required this.landmarks,
  });

  final List<FaceMeshLandmark> landmarks;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.fill;

    for (final lm in landmarks) {
      final double x = lm.x.clamp(0.0, 1.0) * size.width;
      final double y = lm.y.clamp(0.0, 1.0) * size.height;
      canvas.drawCircle(Offset(x, y), 1.0, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _FaceMeshPainter oldDelegate) {
    return !identical(oldDelegate.landmarks, landmarks);
  }
}

class _MyAppState extends State<MyApp> {
  MediapipeFaceMesh? _faceMesh;
  String _status = 'Initializing...';
  FaceMeshResult? _result;
  Uint8List? _sourcePngBytes;
  Uint8List? _displayBytes;
  int? _displayWidth;
  int? _displayHeight;

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
      // Load the bundled PNG asset and convert to RGBA bytes.
      final ByteData data = await rootBundle.load('assets/img.png');
      final Uint8List pngBytes =
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      final img.Image? decoded = img.decodeImage(pngBytes);
      if (decoded == null) {
        throw Exception('Failed to decode PNG asset');
      }
      final img.Image rgba =
          img.copyResize(decoded, width: decoded.width, height: decoded.height);
      final Uint8List rgbaBytes = Uint8List.fromList(
          rgba.convert(numChannels: 4).getBytes(order: img.ChannelOrder.rgba));

      _sourcePngBytes = pngBytes;
      _displayBytes = rgbaBytes;
      _displayWidth = rgba.width;
      _displayHeight = rgba.height;

      final FaceMeshImage image = FaceMeshImage(
        pixels: rgbaBytes,
        width: rgba.width,
        height: rgba.height,
      );
      final FaceMeshResult result = mesh.process(image);
      debugPrint('First 5 landmarks: ${result.landmarks.take(5).map((lm) => '(${lm.x.toStringAsFixed(3)}, ${lm.y.toStringAsFixed(3)}, ${lm.z.toStringAsFixed(3)})').join(', ')}');
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
        body: SafeArea(
          child: SingleChildScrollView(
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
              const SizedBox(height: 12),
              if (_displayBytes != null && _displayWidth != null) ...[
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('Input image'),
                          const SizedBox(height: 8),
                          AspectRatio(
                            aspectRatio: _displayWidth! / (_displayHeight ?? 1),
                            child: Image.memory(
                              _sourcePngBytes ?? _displayBytes!,
                              fit: BoxFit.contain,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('Output with landmarks'),
                          const SizedBox(height: 8),
                          AspectRatio(
                            aspectRatio: _displayWidth! / (_displayHeight ?? 1),
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                Image.memory(
                                  _sourcePngBytes ?? _displayBytes!,
                                  fit: BoxFit.contain,
                                ),
                                if (_result != null)
                                  Positioned.fill(
                                    child: IgnorePointer(
                                      child: CustomPaint(
                                        painter: _FaceMeshPainter(
                                            landmarks: _result!.landmarks),
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
              const SizedBox(height: 16),
              const Text(
                  'Provide a valid mediapipe_face_mesh.tflite and TensorFlow Lite '
                  'runtime libraries to get real predictions.',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
