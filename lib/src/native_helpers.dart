import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart' as pkg_ffi;

/// Rejects a thread count below one. Null means "use the default".
void validateThreads(int? threads) {
  if (threads != null && threads < 1) {
    throw ArgumentError.value(threads, 'threads', 'must be >= 1');
  }
}

/// Default TFLite thread count: half the available cores clamped to 1..4,
/// matching MediaPipe's own CPU inference default.
int defaultInferenceThreads() {
  final int half = Platform.numberOfProcessors ~/ 2;
  if (half < 1) {
    return 1;
  }
  return half > 4 ? 4 : half;
}

/// Reads a native error string; null for both a null pointer and an empty
/// string, because the native error accessors return `std::string::c_str()`
/// of a message that may simply never have been set.
String? readCString(ffi.Pointer<ffi.Char> pointer) {
  if (pointer == ffi.nullptr) {
    return null;
  }
  final String value = pointer.cast<pkg_ffi.Utf8>().toDartString();
  return value.isEmpty ? null : value;
}
