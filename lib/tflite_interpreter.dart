/// Runs an arbitrary TensorFlow Lite model on the runtime this package
/// bundles.
///
/// Experimental: the API may change in any release.
library;

import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart' as pkg_ffi;
import 'package:meta/meta.dart';

import 'mediapipe_face_mesh.dart' show FaceMeshDelegate, FaceMeshException;
import 'src/mediapipe_face_bindings_generated.dart';
import 'src/native_bindings_loader.dart';

/// Element type of a tensor.
@experimental
enum TfliteTensorType {
  /// 32-bit float.
  float32(1, 4),

  /// 32-bit signed integer.
  int32(2, 4),

  /// 8-bit unsigned integer.
  uint8(3, 1),

  /// 64-bit signed integer.
  int64(4, 8),

  /// Boolean, one byte per element.
  bool(6, 1),

  /// 16-bit signed integer.
  int16(7, 2),

  /// 8-bit signed integer.
  int8(9, 1),

  /// 16-bit float.
  float16(10, 2),

  /// 64-bit float.
  float64(11, 8),

  /// 64-bit unsigned integer.
  uint64(13, 8),

  /// 32-bit unsigned integer.
  uint32(16, 4),

  /// 16-bit unsigned integer.
  uint16(17, 2),

  /// Any type this enum does not name, such as string or resource tensors.
  other(-1, 0);

  const TfliteTensorType(this._nativeValue, this.bytesPerElement);

  final int _nativeValue;

  /// Size of one element in bytes. Zero for [other].
  final int bytesPerElement;

  static TfliteTensorType _fromNative(int value) {
    for (final TfliteTensorType type in values) {
      if (type._nativeValue == value) {
        return type;
      }
    }
    return other;
  }
}

/// Name, type, and shape of one input or output tensor.
@experimental
@immutable
class TfliteTensorInfo {
  /// Creates a tensor description.
  const TfliteTensorInfo({
    required this.name,
    required this.type,
    required this.shape,
    required this.byteSize,
  });

  /// Tensor name stored in the model.
  final String name;

  /// Element type.
  final TfliteTensorType type;

  /// Dimensions, outermost first.
  final List<int> shape;

  /// Size of the tensor data in bytes.
  final int byteSize;

  /// Number of elements: the product of [shape].
  int get elementCount =>
      shape.fold<int>(1, (int total, int dim) => total * dim);

  @override
  String toString() =>
      'TfliteTensorInfo($name, ${type.name}, $shape, $byteSize bytes)';
}

final Finalizer<ffi.Pointer<MpInterpreterContext>> _interpreterFinalizer =
    Finalizer<ffi.Pointer<MpInterpreterContext>>(
      (pointer) => faceBindings.mp_interpreter_destroy(pointer),
    );

/// Runs a TensorFlow Lite model on the bundled runtime.
///
/// Tensors keep the shapes stored in the model. Inputs and outputs are flat,
/// in row-major order.
///
/// ```dart
/// final interpreter = TfliteInterpreter.fromBuffer(modelBytes);
/// interpreter.setInput(0, input);
/// interpreter.invoke();
/// final Float32List scores = interpreter.output(0);
/// interpreter.close();
/// ```
///
/// The bundled runtime includes the TensorFlow Lite builtin operators and the
/// MediaPipe custom operators. A model that needs other operators (select
/// TensorFlow ops, for example) fails to load.
///
/// All calls are synchronous. An instance must be used from the isolate that
/// created it.
@experimental
class TfliteInterpreter {
  TfliteInterpreter._(this._context) {
    _interpreterFinalizer.attach(this, _context, detach: this);
    try {
      inputs = List<TfliteTensorInfo>.unmodifiable(<TfliteTensorInfo>[
        for (
          int i = 0;
          i < faceBindings.mp_interpreter_input_count(_context);
          i++
        )
          _readInfo(i, isInput: true),
      ]);
      outputCount = faceBindings.mp_interpreter_output_count(_context);
    } catch (_) {
      close();
      rethrow;
    }
  }

  /// Loads a model from [model] bytes.
  ///
  /// The bytes are copied, so [model] can be discarded after this returns.
  ///
  /// - [delegate] selects CPU or XNNPACK execution; check [activeDelegate]
  ///   for the delegate actually in use.
  /// - [threads] sets the thread count. Defaults to half the CPU cores
  ///   clamped to 1..4.
  /// - [allowDelegateFallback] allows CPU fallback when the requested delegate
  ///   cannot be created. Set it to false to fail creation instead.
  ///
  /// Throws [FaceMeshException] when the model cannot be loaded.
  factory TfliteInterpreter.fromBuffer(
    Uint8List model, {
    int? threads,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    bool allowDelegateFallback = true,
  }) {
    if (model.isEmpty) {
      throw ArgumentError.value(model, 'model', 'must not be empty');
    }
    final ffi.Pointer<ffi.Uint8> modelPtr = pkg_ffi.malloc<ffi.Uint8>(
      model.length,
    );
    try {
      modelPtr.asTypedList(model.length).setAll(0, model);
      return _create(
        threads: threads,
        delegate: delegate,
        allowDelegateFallback: allowDelegateFallback,
        create: (options) => faceBindings.mp_interpreter_create_from_buffer(
          modelPtr,
          model.length,
          options,
        ),
      );
    } finally {
      pkg_ffi.malloc.free(modelPtr);
    }
  }

  /// Loads the model file at [path]. See [TfliteInterpreter.fromBuffer] for
  /// the options.
  factory TfliteInterpreter.fromFile(
    String path, {
    int? threads,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    bool allowDelegateFallback = true,
  }) {
    final ffi.Pointer<pkg_ffi.Utf8> pathPtr = path.toNativeUtf8();
    try {
      return _create(
        threads: threads,
        delegate: delegate,
        allowDelegateFallback: allowDelegateFallback,
        create: (options) => faceBindings.mp_interpreter_create_from_file(
          pathPtr.cast(),
          options,
        ),
      );
    } finally {
      pkg_ffi.malloc.free(pathPtr);
    }
  }

  static TfliteInterpreter _create({
    required int? threads,
    required FaceMeshDelegate delegate,
    required bool allowDelegateFallback,
    required ffi.Pointer<MpInterpreterContext> Function(
      ffi.Pointer<MpInterpreterCreateOptions> options,
    )
    create,
  }) {
    if (threads != null && threads < 1) {
      throw ArgumentError.value(threads, 'threads', 'must be >= 1');
    }
    final optionsPtr = pkg_ffi.calloc<MpInterpreterCreateOptions>();
    try {
      optionsPtr.ref
        ..threads = threads ?? _defaultThreads()
        ..delegate = delegate.index
        ..disable_delegate_fallback = allowDelegateFallback ? 0 : 1;
      final ffi.Pointer<MpInterpreterContext> context = create(optionsPtr);
      if (context == ffi.nullptr) {
        throw FaceMeshException(
          _readCString(faceBindings.mp_interpreter_last_global_error()) ??
              'Failed to create interpreter.',
        );
      }
      return TfliteInterpreter._(context);
    } finally {
      pkg_ffi.calloc.free(optionsPtr);
    }
  }

  final ffi.Pointer<MpInterpreterContext> _context;
  bool _closed = false;

  /// Input tensors, in model order.
  late final List<TfliteTensorInfo> inputs;

  /// Number of output tensors.
  late final int outputCount;

  /// Output tensors, in model order.
  ///
  /// An output with a dynamic shape reports its final shape only after
  /// [invoke].
  List<TfliteTensorInfo> get outputs {
    _ensureNotClosed();
    return <TfliteTensorInfo>[
      for (int i = 0; i < outputCount; i++) _readInfo(i, isInput: false),
    ];
  }

  /// Delegate the model is actually running on after fallback.
  FaceMeshDelegate get activeDelegate {
    _ensureNotClosed();
    return switch (faceBindings.mp_interpreter_active_delegate(_context)) {
      MpDelegateType.MP_DELEGATE_CPU => FaceMeshDelegate.cpu,
      MpDelegateType.MP_DELEGATE_XNNPACK => FaceMeshDelegate.xnnpack,
    };
  }

  /// Writes [data] to the float32 input at [index].
  ///
  /// Throws [ArgumentError] when the input is not float32 or [data] does not
  /// hold exactly its element count.
  void setInput(int index, Float32List data) {
    final TfliteTensorInfo info = _inputInfo(index);
    if (info.type != TfliteTensorType.float32) {
      throw ArgumentError(
        'Input $index is ${info.type.name}, not float32. Use setInputBytes.',
      );
    }
    if (data.lengthInBytes != info.byteSize) {
      throw ArgumentError.value(
        data.length,
        'data.length',
        'input $index holds ${info.byteSize ~/ 4} elements',
      );
    }
    _inputPointer(
      index,
    ).cast<ffi.Float>().asTypedList(data.length).setAll(0, data);
  }

  /// Writes raw [data] to the input at [index], whatever its type.
  ///
  /// Throws [ArgumentError] when [data] is not exactly the tensor's
  /// [TfliteTensorInfo.byteSize].
  void setInputBytes(int index, Uint8List data) {
    final TfliteTensorInfo info = _inputInfo(index);
    if (data.length != info.byteSize) {
      throw ArgumentError.value(
        data.length,
        'data.length',
        'input $index holds ${info.byteSize} bytes',
      );
    }
    _inputPointer(
      index,
    ).cast<ffi.Uint8>().asTypedList(data.length).setAll(0, data);
  }

  /// Runs the model on the current inputs.
  ///
  /// Throws [FaceMeshException] when inference fails.
  void invoke() {
    _ensureNotClosed();
    if (faceBindings.mp_interpreter_invoke(_context) == 0) {
      throw FaceMeshException(_lastError('Interpreter invocation failed.'));
    }
  }

  /// Returns a copy of the float32 output at [index].
  ///
  /// Throws [ArgumentError] when the output is not float32.
  Float32List output(int index) {
    final TfliteTensorInfo info = _outputInfo(index);
    if (info.type != TfliteTensorType.float32) {
      throw ArgumentError(
        'Output $index is ${info.type.name}, not float32. Use outputBytes.',
      );
    }
    final int count = info.byteSize ~/ 4;
    if (count == 0) {
      return Float32List(0);
    }
    return Float32List.fromList(
      _outputPointer(index).cast<ffi.Float>().asTypedList(count),
    );
  }

  /// Returns a copy of the raw bytes of the output at [index], whatever its
  /// type.
  Uint8List outputBytes(int index) {
    final TfliteTensorInfo info = _outputInfo(index);
    if (info.byteSize == 0) {
      return Uint8List(0);
    }
    return Uint8List.fromList(
      _outputPointer(index).cast<ffi.Uint8>().asTypedList(info.byteSize),
    );
  }

  /// Releases the native interpreter and the model it holds.
  void close() {
    if (_closed) {
      return;
    }
    _interpreterFinalizer.detach(this);
    faceBindings.mp_interpreter_destroy(_context);
    _closed = true;
  }

  TfliteTensorInfo _inputInfo(int index) {
    _ensureNotClosed();
    RangeError.checkValidIndex(index, inputs, 'index');
    return inputs[index];
  }

  TfliteTensorInfo _outputInfo(int index) {
    _ensureNotClosed();
    RangeError.checkValidIndex(index, null, 'index', outputCount);
    return _readInfo(index, isInput: false);
  }

  ffi.Pointer<ffi.Void> _inputPointer(int index) {
    final ffi.Pointer<ffi.Void> pointer = faceBindings
        .mp_interpreter_input_data(_context, index);
    if (pointer == ffi.nullptr) {
      throw FaceMeshException(_lastError('Input $index has no data.'));
    }
    return pointer;
  }

  ffi.Pointer<ffi.Void> _outputPointer(int index) {
    final ffi.Pointer<ffi.Void> pointer = faceBindings
        .mp_interpreter_output_data(_context, index);
    if (pointer == ffi.nullptr) {
      throw FaceMeshException(_lastError('Output $index has no data.'));
    }
    return pointer;
  }

  TfliteTensorInfo _readInfo(int index, {required bool isInput}) {
    final ffi.Pointer<MpTensorInfo> infoPtr = pkg_ffi.calloc<MpTensorInfo>();
    try {
      final int ok = isInput
          ? faceBindings.mp_interpreter_input_info(_context, index, infoPtr)
          : faceBindings.mp_interpreter_output_info(_context, index, infoPtr);
      if (ok == 0) {
        throw FaceMeshException(_lastError('Unable to read tensor info.'));
      }
      final MpTensorInfo info = infoPtr.ref;
      return TfliteTensorInfo(
        name: _readCString(info.name) ?? '',
        type: TfliteTensorType._fromNative(info.type),
        shape: List<int>.unmodifiable(<int>[
          for (int i = 0; i < info.dims_count; i++) info.dims[i],
        ]),
        byteSize: info.byte_size,
      );
    } finally {
      pkg_ffi.calloc.free(infoPtr);
    }
  }

  String _lastError(String fallback) =>
      _readCString(faceBindings.mp_interpreter_last_error(_context)) ??
      fallback;

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Interpreter already closed.');
    }
  }
}

/// Half the available cores clamped to 1..4, the same default the face
/// processors use.
int _defaultThreads() {
  final int half = Platform.numberOfProcessors ~/ 2;
  if (half < 1) {
    return 1;
  }
  return half > 4 ? 4 : half;
}

String? _readCString(ffi.Pointer<ffi.Char> pointer) {
  if (pointer == ffi.nullptr) {
    return null;
  }
  final String value = pointer.cast<pkg_ffi.Utf8>().toDartString();
  return value.isEmpty ? null : value;
}
