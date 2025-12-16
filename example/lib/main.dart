import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:mediapipe_face_mesh/mediapipe_face_detection.dart';
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
  MediapipeFaceDetection? _faceDetector;
  String _status = 'Initializing...';
  FaceMeshResult? _result;
  Uint8List? _sourcePngBytes;
  Uint8List? _displayBytes;
  int? _displayWidth;
  int? _displayHeight;
  Uint8List? _cropBytes;
  Uint8List? _detectSourcePngBytes;
  Uint8List? _detectBytes;
  int? _detectWidth;
  int? _detectHeight;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final mesh = await MediapipeFaceMesh.create();
      final detector = await MediapipeFaceDetection.create();
      setState(() {
        _faceMesh = mesh;
        _faceDetector = detector;
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
    final detector = _faceDetector;
    if (mesh == null) {
      return;
    }
    try {
      // Load mesh input image (img.png) and convert to RGBA bytes.
      final ByteData meshData = await rootBundle.load('assets/img.png');
      final Uint8List meshPngBytes = meshData.buffer
          .asUint8List(meshData.offsetInBytes, meshData.lengthInBytes);
      final img.Image? meshDecoded = img.decodeImage(meshPngBytes);
      if (meshDecoded == null) {
        throw Exception('Failed to decode mesh PNG asset');
      }
      final img.Image meshRgba = img.copyResize(meshDecoded,
          width: meshDecoded.width, height: meshDecoded.height);
      final Uint8List meshRgbaBytes = Uint8List.fromList(meshRgba
          .convert(numChannels: 4)
          .getBytes(order: img.ChannelOrder.rgba));

      // Load detection input image (img_2.png) and convert to RGBA bytes.
      Uint8List? detectPngBytes;
      Uint8List? detectRgbaBytes;
      img.Image? detectRgbaImg;
      if (detector != null) {
        final ByteData detectData = await rootBundle.load('assets/img_2.png');
        detectPngBytes = detectData.buffer
            .asUint8List(detectData.offsetInBytes, detectData.lengthInBytes);
        final img.Image? detectDecoded = img.decodeImage(detectPngBytes);
        if (detectDecoded == null) {
          throw Exception('Failed to decode detection PNG asset');
        }
        detectRgbaImg = img.copyResize(detectDecoded,
            width: detectDecoded.width, height: detectDecoded.height);
        detectRgbaBytes = Uint8List.fromList(detectRgbaImg
            .convert(numChannels: 4)
            .getBytes(order: img.ChannelOrder.rgba));
      }

      _sourcePngBytes = meshPngBytes;
      _displayBytes = meshRgbaBytes;
      _displayWidth = meshRgba.width;
      _displayHeight = meshRgba.height;
      _detectSourcePngBytes = detectPngBytes;
      _detectBytes = detectRgbaBytes;
      _detectWidth = detectRgbaImg?.width;
      _detectHeight = detectRgbaImg?.height;

      final FaceMeshImage image = FaceMeshImage(
        pixels: meshRgbaBytes,
        width: meshRgba.width,
        height: meshRgba.height,
      );
      FaceDetectionResult? detectionResult;
      Uint8List? cropBytes;
      if (detector != null && detectRgbaBytes != null && detectRgbaImg != null) {
        final FaceMeshImage detectImage = FaceMeshImage(
          pixels: detectRgbaBytes,
          width: detectRgbaImg.width,
          height: detectRgbaImg.height,
        );
        detectionResult = detector.process(detectImage);
        if (detectionResult.detections.isNotEmpty) {
          final detection = detectionResult.detections.first;
          final rect = detection.rect;
          final int cropWidth = (rect.width * detectRgbaImg.width)
              .clamp(1.0, detectRgbaImg.width.toDouble())
              .toInt();
          final int cropHeight = (rect.height * detectRgbaImg.height)
              .clamp(1.0, detectRgbaImg.height.toDouble())
              .toInt();
          final int left =
              ((rect.xCenter - rect.width / 2) * detectRgbaImg.width)
              .clamp(0.0, (detectRgbaImg.width - 1).toDouble())
              .toInt();
          final int top =
              ((rect.yCenter - rect.height / 2) * detectRgbaImg.height)
              .clamp(0.0, (detectRgbaImg.height - 1).toDouble())
              .toInt();
          final int w = math.min(cropWidth, detectRgbaImg.width - left);
          final int h = math.min(cropHeight, detectRgbaImg.height - top);
          final img.Image cropped = img.copyCrop(detectRgbaImg,
              x: left, y: top, width: w, height: h);
          cropBytes = Uint8List.fromList(img.encodePng(cropped));

          debugPrint(
              '[FaceDetection]'
                  ' rect(xCenter=${rect.xCenter.toStringAsFixed(4)},'
                  ' yCenter=${rect.yCenter.toStringAsFixed(4)},'
                  ' w=${rect.width.toStringAsFixed(4)},'
                  ' h=${rect.height.toStringAsFixed(4)})'
          );
        }
      }
      final FaceMeshResult result = mesh.process(image);
      debugPrint('First 5 landmarks: ${result.landmarks.take(5).map((lm) => '(${lm.x.toStringAsFixed(3)}, ${lm.y.toStringAsFixed(3)}, ${lm.z.toStringAsFixed(3)})').join(', ')}');
      setState(() {
        _result = result;
        _cropBytes = cropBytes;
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
    _faceDetector?.close();
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
              if (_displayBytes != null && _displayWidth != null) ...[
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('Detect input'),
                          const SizedBox(height: 8),
                          AspectRatio(
                            aspectRatio: _displayWidth! / (_displayHeight ?? 1),
                            child: Image.memory(
                              _detectSourcePngBytes ?? _displayBytes!,
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
                          const Text('Detection crop'),
                          const SizedBox(height: 8),
                          AspectRatio(
                            aspectRatio: _displayWidth! / (_displayHeight ?? 1),
                            child: _cropBytes != null
                                ? Image.memory(
                                    _cropBytes!,
                                    fit: BoxFit.contain,
                                  )
                                : Container(
                                    color: Colors.grey.shade200,
                                    alignment: Alignment.center,
                                    child: const Text('No detection'),
                                  ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
              ],
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
