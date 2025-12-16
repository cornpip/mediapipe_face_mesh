part of '../mediapipe_face_mesh.dart';

class _NativeImage {
  _NativeImage({
    required this.image,
    required this.pixels,
  });

  final ffi.Pointer<MpImage> image;
  final ffi.Pointer<ffi.Uint8> pixels;
}

_NativeImage _toNativeImage(FaceMeshImage image) {
  final ffi.Pointer<MpImage> imagePtr = pkg_ffi.calloc<MpImage>();
  final ffi.Pointer<ffi.Uint8> pixelPtr =
      pkg_ffi.calloc<ffi.Uint8>(image.pixels.length);
  pixelPtr.asTypedList(image.pixels.length).setAll(0, image.pixels);
  imagePtr.ref
    ..data = pixelPtr.cast()
    ..width = image.width
    ..height = image.height
    ..bytes_per_row = image.bytesPerRow
    ..format = image.pixelFormat;
  return _NativeImage(image: imagePtr, pixels: pixelPtr);
}

ffi.Pointer<MpNormalizedRect> _toNativeRect(NormalizedRect rect) {
  final ffi.Pointer<MpNormalizedRect> roiPtr = pkg_ffi.calloc<MpNormalizedRect>();
  roiPtr.ref
    ..x_center = rect.xCenter
    ..y_center = rect.yCenter
    ..width = rect.width
    ..height = rect.height
    ..rotation = rect.rotation;
  return roiPtr;
}

String? _readCString(ffi.Pointer<ffi.Char> pointer) {
  if (pointer == ffi.nullptr) {
    return null;
  }
  return pointer.cast<pkg_ffi.Utf8>().toDartString();
}
