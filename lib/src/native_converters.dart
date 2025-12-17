part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

class _NativeImage {
  _NativeImage({
    required this.image,
    required this.pixels,
  });

  final ffi.Pointer<MpImage> image;
  final ffi.Pointer<ffi.Uint8> pixels;
}

class _NativeNv21Image {
  _NativeNv21Image({
    required this.image,
    required this.yPlane,
    required this.vuPlane,
  });

  final ffi.Pointer<MpNv21Image> image;
  final ffi.Pointer<ffi.Uint8> yPlane;
  final ffi.Pointer<ffi.Uint8> vuPlane;
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

_NativeNv21Image _toNativeNv21Image(FaceMeshNv21Image image) {
  final ffi.Pointer<MpNv21Image> imagePtr = pkg_ffi.calloc<MpNv21Image>();
  final ffi.Pointer<ffi.Uint8> yPtr =
      pkg_ffi.calloc<ffi.Uint8>(image.yPlane.length);
  yPtr.asTypedList(image.yPlane.length).setAll(0, image.yPlane);
  final ffi.Pointer<ffi.Uint8> vuPtr =
      pkg_ffi.calloc<ffi.Uint8>(image.vuPlane.length);
  vuPtr.asTypedList(image.vuPlane.length).setAll(0, image.vuPlane);
  imagePtr.ref
    ..y = yPtr
    ..vu = vuPtr
    ..width = image.width
    ..height = image.height
    ..y_bytes_per_row = image.yBytesPerRow
    ..vu_bytes_per_row = image.vuBytesPerRow;
  return _NativeNv21Image(image: imagePtr, yPlane: yPtr, vuPlane: vuPtr);
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
