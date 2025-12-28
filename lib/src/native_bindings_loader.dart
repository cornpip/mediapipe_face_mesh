import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:mediapipe_face_mesh/src/mediapipe_face_bindings_generated.dart';

MediapipeFaceBindings? _faceBindings;
ffi.DynamicLibrary? _faceDylib;

/// Optionally initialize bindings with a specific dynamic library or path.
/// If not called, the first access will lazily load the default library.
void initializeFaceBindings({ffi.DynamicLibrary? dylib, String? libraryPath}) {
  _faceDylib = dylib ?? _openDefaultDylib(libraryPath: libraryPath);
  _faceBindings = MediapipeFaceBindings(_faceDylib!);
}

/// Returns a lazily initialized bindings instance for the native library.
MediapipeFaceBindings get faceBindings =>
    _faceBindings ??= MediapipeFaceBindings(_faceDylib ??= _openDefaultDylib());

/// Returns the dynamic library handle used to resolve native symbols.
ffi.DynamicLibrary get faceDylib => _faceDylib ??= _openDefaultDylib();

ffi.DynamicLibrary _openDefaultDylib({String? libraryPath}) {
  if (libraryPath != null && libraryPath.isNotEmpty) {
    return ffi.DynamicLibrary.open(libraryPath);
  }
  const String libName = 'mediapipe_face_mesh';
  if (Platform.isMacOS || Platform.isIOS) {
    return ffi.DynamicLibrary.open('$libName.framework/$libName');
  }
  if (Platform.isAndroid || Platform.isLinux) {
    return ffi.DynamicLibrary.open('lib$libName.so');
  }
  if (Platform.isWindows) {
    return ffi.DynamicLibrary.open('$libName.dll');
  }
  throw UnsupportedError('Unsupported platform ${Platform.operatingSystem}');
}
