part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Reusable native buffers for uploading one frame per process call.
///
/// Each processor owns one instance. Buffers grow on demand and stay
/// allocated between calls, so the steady-state cost per call is a single
/// memcpy instead of an allocate + zero-fill + copy + free cycle. The native
/// layer only reads the buffers during the synchronous process call, so
/// reuse across calls is safe. Released by the processor's close() or its
/// finalizer.
class _FrameScratch {
  final ffi.Pointer<MpImage> _imagePtr = pkg_ffi.calloc<MpImage>();
  final ffi.Pointer<MpNv21Image> _nv21Ptr = pkg_ffi.calloc<MpNv21Image>();
  ffi.Pointer<ffi.Uint8> _primary = ffi.nullptr;
  int _primaryCapacity = 0;
  ffi.Pointer<ffi.Uint8> _secondary = ffi.nullptr;
  int _secondaryCapacity = 0;
  bool _disposed = false;

  ffi.Pointer<ffi.Uint8> _ensurePrimary(int size) {
    if (size > _primaryCapacity) {
      if (_primary != ffi.nullptr) {
        pkg_ffi.malloc.free(_primary);
      }
      _primary = pkg_ffi.malloc<ffi.Uint8>(size);
      _primaryCapacity = size;
    }
    return _primary;
  }

  ffi.Pointer<ffi.Uint8> _ensureSecondary(int size) {
    if (size > _secondaryCapacity) {
      if (_secondary != ffi.nullptr) {
        pkg_ffi.malloc.free(_secondary);
      }
      _secondary = pkg_ffi.malloc<ffi.Uint8>(size);
      _secondaryCapacity = size;
    }
    return _secondary;
  }

  /// Copies [image] into reused native memory and returns the frame struct.
  /// The returned pointer stays valid until the next call or [dispose].
  ffi.Pointer<MpImage> imageFrom(FaceMeshImage image) {
    assert(!_disposed);
    final ffi.Pointer<ffi.Uint8> pixels = _ensurePrimary(image.pixels.length);
    pixels.asTypedList(image.pixels.length).setAll(0, image.pixels);
    _imagePtr.ref
      ..data = pixels.cast()
      ..width = image.width
      ..height = image.height
      ..bytes_per_row = image.bytesPerRow
      ..format = image.pixelFormat;
    return _imagePtr;
  }

  /// NV21 variant of [imageFrom]; Y and VU planes use separate buffers.
  ffi.Pointer<MpNv21Image> nv21From(FaceMeshNv21Image image) {
    assert(!_disposed);
    final ffi.Pointer<ffi.Uint8> yPtr = _ensurePrimary(image.yPlane.length);
    yPtr.asTypedList(image.yPlane.length).setAll(0, image.yPlane);
    final ffi.Pointer<ffi.Uint8> vuPtr = _ensureSecondary(
      image.vuPlane.length,
    );
    vuPtr.asTypedList(image.vuPlane.length).setAll(0, image.vuPlane);
    _nv21Ptr.ref
      ..y = yPtr
      ..vu = vuPtr
      ..width = image.width
      ..height = image.height
      ..y_bytes_per_row = image.yBytesPerRow
      ..vu_bytes_per_row = image.vuBytesPerRow;
    return _nv21Ptr;
  }

  void dispose() {
    if (_disposed) {
      return;
    }
    _disposed = true;
    if (_primary != ffi.nullptr) {
      pkg_ffi.malloc.free(_primary);
      _primary = ffi.nullptr;
      _primaryCapacity = 0;
    }
    if (_secondary != ffi.nullptr) {
      pkg_ffi.malloc.free(_secondary);
      _secondary = ffi.nullptr;
      _secondaryCapacity = 0;
    }
    pkg_ffi.calloc.free(_imagePtr);
    pkg_ffi.calloc.free(_nv21Ptr);
  }
}

final Finalizer<_FrameScratch> _frameScratchFinalizer = Finalizer(
  (_FrameScratch scratch) => scratch.dispose(),
);

ffi.Pointer<MpNormalizedRect> _toNativeRect(NormalizedRect rect) {
  final ffi.Pointer<MpNormalizedRect> roiPtr = pkg_ffi
      .calloc<MpNormalizedRect>();
  roiPtr.ref
    ..x_center = rect.xCenter
    ..y_center = rect.yCenter
    ..width = rect.width
    ..height = rect.height
    ..rotation = rect.rotation;
  return roiPtr;
}

ffi.Pointer<MpNormalizedRect> _toNativeRectArray(List<NormalizedRect> rects) {
  final ffi.Pointer<MpNormalizedRect> rectsPtr = pkg_ffi
      .calloc<MpNormalizedRect>(rects.length);
  for (int i = 0; i < rects.length; i++) {
    (rectsPtr + i).ref
      ..x_center = rects[i].xCenter
      ..y_center = rects[i].yCenter
      ..width = rects[i].width
      ..height = rects[i].height
      ..rotation = rects[i].rotation;
  }
  return rectsPtr;
}

String? _readCString(ffi.Pointer<ffi.Char> pointer) {
  if (pointer == ffi.nullptr) {
    return null;
  }
  return pointer.cast<pkg_ffi.Utf8>().toDartString();
}
