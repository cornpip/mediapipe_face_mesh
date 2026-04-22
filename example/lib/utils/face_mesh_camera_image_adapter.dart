import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Example-only adapter helpers for converting `camera` frames into package
/// input types without adding `camera` as a dependency of the core package.
class FaceMeshCameraImageAdapter {
  const FaceMeshCameraImageAdapter._();

  static FaceMeshNv21Image? toNv21(CameraImage image) {
    final planes = image.planes;
    if (planes.isEmpty) {
      return null;
    }

    Uint8List yPlane;
    Uint8List vuPlane;
    int yBytesPerRow;
    int vuBytesPerRow;

    if (planes.length >= 2) {
      yPlane = planes[0].bytes;
      vuPlane = planes[1].bytes;
      yBytesPerRow = planes[0].bytesPerRow;
      vuBytesPerRow = planes[1].bytesPerRow;
    } else {
      final plane = planes.first;
      final rowStride = plane.bytesPerRow;
      final ySize = rowStride * image.height;
      final vuSize = rowStride * ((image.height + 1) ~/ 2);
      if (plane.bytes.length < ySize + vuSize) {
        return null;
      }
      yPlane = Uint8List.sublistView(plane.bytes, 0, ySize);
      vuPlane = Uint8List.sublistView(plane.bytes, ySize, ySize + vuSize);
      yBytesPerRow = rowStride;
      vuBytesPerRow = rowStride;
    }

    return FaceMeshNv21Image(
      yPlane: yPlane,
      vuPlane: vuPlane,
      width: image.width,
      height: image.height,
      yBytesPerRow: yBytesPerRow,
      vuBytesPerRow: vuBytesPerRow,
    );
  }

  static FaceMeshImage? toBgra(CameraImage image) {
    final planes = image.planes;
    if (planes.isEmpty) {
      return null;
    }
    final plane = planes.first;
    return FaceMeshImage(
      pixels: plane.bytes,
      width: image.width,
      height: image.height,
      bytesPerRow: plane.bytesPerRow,
      pixelFormat: FaceMeshPixelFormat.bgra,
    );
  }
}
