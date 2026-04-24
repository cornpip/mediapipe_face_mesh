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

    if ((image.width & 1) != 0 || (image.height & 1) != 0) {
      return null;
    }
    if (planes.length == 1) {
      return _fromSinglePlaneNv21(image);
    }
    if (planes.length >= 3) {
      return _fromYuv420Planes(image);
    }
    if (planes.length == 2) {
      return _fromYAndInterleavedVuPlanes(image);
    }
    return null;
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

  static FaceMeshNv21Image? _fromSinglePlaneNv21(CameraImage image) {
    final plane = image.planes.first;
    final rowStride = plane.bytesPerRow;
    final ySize = rowStride * image.height;
    final vuSize = rowStride * (image.height ~/ 2);
    if (plane.bytes.length < ySize + vuSize) {
      return null;
    }
    return FaceMeshNv21Image(
      yPlane: Uint8List.sublistView(plane.bytes, 0, ySize),
      vuPlane: Uint8List.sublistView(plane.bytes, ySize, ySize + vuSize),
      width: image.width,
      height: image.height,
      yBytesPerRow: rowStride,
      vuBytesPerRow: rowStride,
    );
  }

  static FaceMeshNv21Image? _fromYAndInterleavedVuPlanes(CameraImage image) {
    final yPlane = _copyPlane(
      image.planes[0],
      width: image.width,
      height: image.height,
    );
    final vuPlane = _copyPlane(
      image.planes[1],
      width: image.width,
      height: image.height ~/ 2,
    );
    if (yPlane == null || vuPlane == null) {
      return null;
    }
    return FaceMeshNv21Image(
      yPlane: yPlane,
      vuPlane: vuPlane,
      width: image.width,
      height: image.height,
      yBytesPerRow: image.width,
      vuBytesPerRow: image.width,
    );
  }

  static FaceMeshNv21Image? _fromYuv420Planes(CameraImage image) {
    final yPlane = _copyPlane(
      image.planes[0],
      width: image.width,
      height: image.height,
    );
    if (yPlane == null) {
      return null;
    }

    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final uvWidth = image.width ~/ 2;
    final uvHeight = image.height ~/ 2;
    final vuPlane = Uint8List(image.width * uvHeight);

    for (var row = 0; row < uvHeight; row++) {
      for (var col = 0; col < uvWidth; col++) {
        final u = _readPlaneByte(uPlane, row, col);
        final v = _readPlaneByte(vPlane, row, col);
        if (u == null || v == null) {
          return null;
        }
        final out = row * image.width + col * 2;
        vuPlane[out] = v;
        vuPlane[out + 1] = u;
      }
    }

    return FaceMeshNv21Image(
      yPlane: yPlane,
      vuPlane: vuPlane,
      width: image.width,
      height: image.height,
      yBytesPerRow: image.width,
      vuBytesPerRow: image.width,
    );
  }

  static Uint8List? _copyPlane(
    Plane plane, {
    required int width,
    required int height,
  }) {
    final out = Uint8List(width * height);
    for (var row = 0; row < height; row++) {
      for (var col = 0; col < width; col++) {
        final value = _readPlaneByte(plane, row, col);
        if (value == null) {
          return null;
        }
        out[row * width + col] = value;
      }
    }
    return out;
  }

  static int? _readPlaneByte(Plane plane, int row, int col) {
    final pixelStride = plane.bytesPerPixel ?? 1;
    final index = row * plane.bytesPerRow + col * pixelStride;
    if (index < 0 || index >= plane.bytes.length) {
      return null;
    }
    return plane.bytes[index];
  }
}
