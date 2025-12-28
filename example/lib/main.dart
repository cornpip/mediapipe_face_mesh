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
  _FaceMeshPainter({required this.landmarks});

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
  FaceMeshProcessor? _faceMeshProcessor;
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
      final mesh = await FaceMeshProcessor.create(
        delegate: FaceMeshDelegate.xnnpack,
      );
      setState(() {
        _faceMeshProcessor = mesh;
        _status = 'Engine ready. Tap the button to run a dummy inference.';
      });
    } catch (error) {
      setState(() {
        _status =
            'Initialization failed (expected until a proper TFLite runtime/model is bundled): $error';
      });
      debugPrint(error.toString());
    }
  }

  Future<void> _runOnce() async {
    final mesh = _faceMeshProcessor;
    if (mesh == null) {
      return;
    }
    try {
      // Load the PNG asset into memory (compressed PNG bytes),
      // decode it to an image so pixels are available,
      // then re-encode those pixels into raw RGBA bytes for the native API.
      final ByteData byteData = await rootBundle.load('assets/img.png');
      final Uint8List pngByte = byteData.buffer.asUint8List(
        byteData.offsetInBytes,
        byteData.lengthInBytes,
      );
      final img.Image? decodedData = img.decodeImage(pngByte);
      if (decodedData == null) {
        throw Exception('Failed to decode mesh PNG asset');
      }
      final img.Image rgba = img.copyResize(
        decodedData,
        width: decodedData.width,
        height: decodedData.height,
      );
      final Uint8List rgbaBytes = Uint8List.fromList(
        rgba.convert(numChannels: 4).getBytes(order: img.ChannelOrder.rgba),
      );

      _sourcePngBytes = pngByte;
      _displayBytes = rgbaBytes;
      _displayWidth = rgba.width;
      _displayHeight = rgba.height;

      final FaceMeshImage image = FaceMeshImage(
        pixels: rgbaBytes,
        width: rgba.width,
        height: rgba.height,
      );
      final FaceMeshResult result = mesh.process(image);
      debugPrint(
        'First 5 landmarks: ${result.landmarks.take(5).map((lm) => '(${lm.x.toStringAsFixed(3)}, ${lm.y.toStringAsFixed(3)}, ${lm.z.toStringAsFixed(3)})').join(', ')}',
      );
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
    _faceMeshProcessor?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('MediaPipe Face Mesh')),
        body: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(_status, style: Theme.of(context).textTheme.bodyLarge),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _faceMeshProcessor == null ? null : _runOnce,
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
                              aspectRatio:
                                  _displayWidth! / (_displayHeight ?? 1),
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
                              aspectRatio:
                                  _displayWidth! / (_displayHeight ?? 1),
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
                                            landmarks: _result!.landmarks,
                                          ),
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
