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
      final plane = planes.first;
      return FaceMeshNv21Image.tryFromSinglePlane(
        bytes: plane.bytes,
        width: image.width,
        height: image.height,
        bytesPerRow: plane.bytesPerRow,
      );
    }
    if (planes.length == 2) {
      return FaceMeshNv21Image.tryFromYAndInterleavedVuPlanes(
        width: image.width,
        height: image.height,
        yPlane: planes[0].toFaceMeshImagePlane(),
        vuPlane: planes[1].toFaceMeshImagePlane(),
      );
    }
    if (planes.length >= 3) {
      return FaceMeshNv21Image.tryFromYuv420Planes(
        width: image.width,
        height: image.height,
        yPlane: planes[0].toFaceMeshImagePlane(),
        uPlane: planes[1].toFaceMeshImagePlane(),
        vPlane: planes[2].toFaceMeshImagePlane(),
      );
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
}

extension on Plane {
  FaceMeshImagePlane toFaceMeshImagePlane() {
    return FaceMeshImagePlane(
      bytes: bytes,
      bytesPerRow: bytesPerRow,
      bytesPerPixel: bytesPerPixel,
    );
  }
}
