part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

Future<String> _materializeModel() async {
  const String key = _defaultModelAsset;
  final ByteData data = await rootBundle.load(key);
  final Directory cacheDir =
      Directory('${Directory.systemTemp.path}/mediapipe_face_mesh_cache');
  if (!await cacheDir.exists()) {
    await cacheDir.create(recursive: true);
  }
  final String sanitizedName =
      key.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '_');
  final File file = File('${cacheDir.path}/$sanitizedName');
  final List<int> bytes =
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  if (!await file.exists() || await file.length() != bytes.length) {
    await file.writeAsBytes(
      bytes,
      flush: true,
    );
  }
  return file.path;
}

NormalizedRect _normalizedRectFromBox(
  FaceMeshBox box, {
  required int imageWidth,
  required int imageHeight,
  required double scale,
  bool makeSquare = true,
}) {
  if (imageWidth <= 0 || imageHeight <= 0) {
    throw ArgumentError('Invalid image size: ${imageWidth}x$imageHeight');
  }
  if (!(scale > 0)) {
    throw ArgumentError('scale must be > 0.');
  }
  if (!(box.right > box.left) || !(box.bottom > box.top)) {
    throw ArgumentError('Invalid box: left/top must be < right/bottom.');
  }

  final double clampedLeft =
      box.left.clamp(0.0, imageWidth.toDouble()).toDouble();
  final double clampedTop =
      box.top.clamp(0.0, imageHeight.toDouble()).toDouble();
  final double clampedRight =
      box.right.clamp(0.0, imageWidth.toDouble()).toDouble();
  final double clampedBottom =
      box.bottom.clamp(0.0, imageHeight.toDouble()).toDouble();

  final double centerX = (clampedLeft + clampedRight) * 0.5;
  final double centerY = (clampedTop + clampedBottom) * 0.5;

  double width = (clampedRight - clampedLeft).abs();
  double height = (clampedBottom - clampedTop).abs();

  if (makeSquare) {
    final double size = width > height ? width : height;
    width = size;
    height = size;
  }

  width *= scale;
  height *= scale;

  return NormalizedRect(
    xCenter: centerX / imageWidth,
    yCenter: centerY / imageHeight,
    width: width / imageWidth,
    height: height / imageHeight,
    rotation: 0,
  );
}
