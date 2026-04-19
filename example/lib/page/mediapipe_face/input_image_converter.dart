import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class InputImageConverter {
  InputImage? fromCameraImage({
    required CameraImage image,
    required CameraController controller,
    required CameraDescription camera,
    required InputImageRotation inputImageRotation,
  }) {
    final format = InputImageFormatValue.fromRawValue(image.format.raw);

    final isValidFormat = format != null &&
        ((Platform.isAndroid && format == InputImageFormat.nv21) ||
            (Platform.isIOS && format == InputImageFormat.bgra8888));

    if (!isValidFormat) return null;
    if (image.planes.length != 1) return null;

    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: inputImageRotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }
}
