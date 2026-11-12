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

  /// 16-bit float. Readable and writable as raw bytes only.
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

/// Quantization parameters of a tensor:
/// `real = (quantized - zeroPoint) * scale`.
@experimental
@immutable
class TfliteQuantization {
  /// Creates quantization parameters.
  const TfliteQuantization({required this.scale, required this.zeroPoint});

  /// Step between two adjacent quantized values.
  final double scale;

  /// Quantized value that represents real zero.
  final int zeroPoint;

  /// Converts a quantized value to its real value.
  double dequantize(int value) => (value - zeroPoint) * scale;

  /// Converts a real value to the nearest quantized value, without clamping
  /// it to the tensor type's range.
  int quantize(double value) => (value / scale).round() + zeroPoint;

  @override
  String toString() =>
      'TfliteQuantization(scale: $scale, zeroPoint: $zeroPoint)';
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
    this.quantization,
  });

  /// Tensor name stored in the model.
  final String name;

  /// Element type.
  final TfliteTensorType type;

  /// Dimensions, outermost first.
  final List<int> shape;

  /// Size of the tensor data in bytes.
  final int byteSize;

  /// Quantization parameters, or null when the tensor is not quantized.
  final TfliteQuantization? quantization;

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
/// [run] takes and fills nested lists shaped like the tensors:
///
/// ```dart
/// final interpreter = TfliteInterpreter.fromBuffer(modelBytes);
/// final output = [List<double>.filled(2, 0)]; // shape [1, 2]
/// interpreter.run(image, output); // image: nested list or typed data
/// interpreter.close();
/// ```
///
/// [setInput], [invoke], and [output] do the same with flat typed data and
/// skip the list conversion:
///
/// ```dart
/// interpreter.setInput(0, input); // Float32List, row-major
/// interpreter.invoke();
/// final Float32List scores = interpreter.output(0);
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
      outputCount = faceBindings.mp_interpreter_output_count(_context);
      _inputs = _readInputs();
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

  /// True between [resizeInput] and the next allocation.
  bool _needsAllocation = false;
  List<TfliteTensorInfo>? _inputs;

  /// Number of output tensors.
  late final int outputCount;

  /// Input tensors, in model order.
  List<TfliteTensorInfo> get inputs {
    _ensureReady();
    return _inputs ??= _readInputs();
  }

  /// Output tensors, in model order.
  ///
  /// An output with a dynamic shape reports its final shape only after
  /// [invoke].
  List<TfliteTensorInfo> get outputs {
    _ensureReady();
    return <TfliteTensorInfo>[
      for (int i = 0; i < outputCount; i++) _readInfo(i, isInput: false),
    ];
  }

  /// Index of the input tensor named [name].
  ///
  /// Throws [ArgumentError] when no input has that name.
  int inputIndex(String name) => _indexOf(inputs, name, 'input');

  /// Index of the output tensor named [name].
  ///
  /// Throws [ArgumentError] when no output has that name.
  int outputIndex(String name) => _indexOf(outputs, name, 'output');

  /// Delegate the model is actually running on after fallback.
  FaceMeshDelegate get activeDelegate {
    _ensureNotClosed();
    return switch (faceBindings.mp_interpreter_active_delegate(_context)) {
      MpDelegateType.MP_DELEGATE_CPU => FaceMeshDelegate.cpu,
      MpDelegateType.MP_DELEGATE_XNNPACK => FaceMeshDelegate.xnnpack,
    };
  }

  /// Changes the shape of the input at [index] to [shape].
  ///
  /// Tensors are reallocated on the next call that needs them, or right away
  /// with [allocateTensors]. Input data written before the resize is lost.
  void resizeInput(int index, List<int> shape) {
    _ensureNotClosed();
    RangeError.checkValidIndex(index, null, 'index', _inputCount);
    if (shape.length > MP_TENSOR_MAX_DIMS) {
      throw ArgumentError.value(
        shape,
        'shape',
        'at most $MP_TENSOR_MAX_DIMS dimensions are supported',
      );
    }
    final ffi.Pointer<ffi.Int32> dimsPtr = pkg_ffi.calloc<ffi.Int32>(
      shape.isEmpty ? 1 : shape.length,
    );
    try {
      dimsPtr.asTypedList(shape.length).setAll(0, shape);
      if (faceBindings.mp_interpreter_resize_input(
            _context,
            index,
            dimsPtr,
            shape.length,
          ) ==
          0) {
        throw FaceMeshException(_lastError('Unable to resize input $index.'));
      }
    } finally {
      pkg_ffi.calloc.free(dimsPtr);
    }
    _needsAllocation = true;
    _inputs = null;
  }

  /// Reallocates tensors after [resizeInput].
  ///
  /// Throws [FaceMeshException] when the model cannot run with the new
  /// shapes.
  void allocateTensors() {
    _ensureNotClosed();
    if (faceBindings.mp_interpreter_allocate_tensors(_context) == 0) {
      throw FaceMeshException(_lastError('Tensor allocation failed.'));
    }
    _needsAllocation = false;
    _inputs = null;
  }

  /// Runs the model on one [input] and writes its first output to [output].
  ///
  /// See [runForMultipleInputs] for the accepted values.
  void run(Object input, Object output) =>
      runForMultipleInputs(<Object>[input], <int, Object>{0: output});

  /// Runs the model on [inputs], in model order, and writes the outputs
  /// named by index in [outputs].
  ///
  /// Each input is a nested list shaped like its tensor, a flat list, a
  /// typed list of the tensor's type, or raw bytes ([Uint8List], [ByteData],
  /// [ByteBuffer]). A nested list whose shape differs from the tensor's
  /// resizes that input first.
  ///
  /// Each output is a nested list shaped like its tensor, filled in place, or
  /// typed data of the tensor's byte size.
  ///
  /// Throws [ArgumentError] when a value does not fit its tensor.
  void runForMultipleInputs(List<Object> inputs, Map<int, Object> outputs) {
    _ensureNotClosed();
    if (inputs.length != _inputCount) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'the model has $_inputCount inputs',
      );
    }
    for (int i = 0; i < inputs.length; i++) {
      final Object input = inputs[i];
      if (input is TypedData || input is ByteBuffer || input is! List) {
        continue;
      }
      final List<int> shape = _nestedShape(input);
      final TfliteTensorInfo info = this.inputs[i];
      final int count = shape.fold<int>(1, (int a, int b) => a * b);
      if (count != info.elementCount) {
        resizeInput(i, shape);
      }
    }
    for (int i = 0; i < inputs.length; i++) {
      setInput(i, inputs[i]);
    }
    invoke();
    outputs.forEach(copyOutputTo);
  }

  /// Writes [data] to the input at [index].
  ///
  /// [data] is a typed list of the tensor's type (a [Float32List] for a
  /// float32 tensor), raw bytes ([Uint8List], [ByteData], [ByteBuffer]), or a
  /// flat or nested list of numbers or bools in row-major order.
  ///
  /// Throws [ArgumentError] when [data] does not match the tensor's type or
  /// size.
  void setInput(int index, Object data) {
    final TfliteTensorInfo info = _inputInfo(index);
    if (data is ByteBuffer) {
      _writeBytes(index, info, data.asUint8List());
    } else if (data is TypedData) {
      if (data is! Uint8List &&
          data is! ByteData &&
          !_isTypedListOf(data, info.type)) {
        throw ArgumentError(
          'Input $index is ${info.type.name}, but the data is a '
          '${data.runtimeType}.',
        );
      }
      _writeBytes(
        index,
        info,
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      );
    } else if (data is List) {
      final List<num> view = _numView(
        _inputPointer(index),
        info.type,
        info.elementCount,
        'Input $index',
      );
      final int written = _flattenInto(data, view, 0, info.type);
      if (written != info.elementCount) {
        throw ArgumentError(
          'Input $index holds ${info.elementCount} elements, got $written.',
        );
      }
    } else {
      throw ArgumentError.value(
        data,
        'data',
        'must be a list, typed data, or a byte buffer',
      );
    }
  }

  /// Writes raw [data] to the input at [index], whatever its type.
  ///
  /// Throws [ArgumentError] when [data] is not exactly the tensor's
  /// [TfliteTensorInfo.byteSize].
  void setInputBytes(int index, Uint8List data) =>
      _writeBytes(index, _inputInfo(index), data);

  /// Runs the model on the current inputs.
  ///
  /// Throws [FaceMeshException] when inference fails.
  void invoke() {
    _ensureReady();
    if (faceBindings.mp_interpreter_invoke(_context) == 0) {
      throw FaceMeshException(_lastError('Interpreter invocation failed.'));
    }
  }

  /// Returns a copy of the float32 output at [index].
  ///
  /// Throws [ArgumentError] when the output is not float32. Use [outputData]
  /// for other types.
  Float32List output(int index) {
    final TfliteTensorInfo info = _outputInfo(index);
    if (info.type != TfliteTensorType.float32) {
      throw ArgumentError(
        'Output $index is ${info.type.name}, not float32. Use outputData.',
      );
    }
    return outputData(index) as Float32List;
  }

  /// Returns a copy of the output at [index] as a typed list of its type: a
  /// [Float32List] for float32, an [Int32List] for int32, and so on. Bool
  /// tensors come back as a [Uint8List] of 0 and 1.
  ///
  /// Throws [UnsupportedError] for float16 and [TfliteTensorType.other]
  /// tensors. Use [outputBytes] for those.
  TypedData outputData(int index) {
    final TfliteTensorInfo info = _outputInfo(index);
    final int count = info.elementCount;
    if (count == 0) {
      return _emptyTyped(info.type, 'Output $index');
    }
    return _copyTyped(
      _numView(_outputPointer(index), info.type, count, 'Output $index'),
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

  /// Copies the output at [index] into [target].
  ///
  /// [target] is a nested list shaped like the tensor, whose innermost lists
  /// are filled in place, or typed data of the tensor's byte size.
  ///
  /// Throws [ArgumentError] when [target] does not fit the tensor.
  void copyOutputTo(int index, Object target) {
    final TfliteTensorInfo info = _outputInfo(index);
    if (target is ByteBuffer || target is TypedData) {
      final Uint8List bytes = target is ByteBuffer
          ? target.asUint8List()
          : (target as TypedData).buffer.asUint8List(
              target.offsetInBytes,
              target.lengthInBytes,
            );
      if (bytes.length != info.byteSize) {
        throw ArgumentError(
          'Output $index holds ${info.byteSize} bytes, the target holds '
          '${bytes.length}.',
        );
      }
      if (info.byteSize > 0) {
        bytes.setAll(
          0,
          _outputPointer(index).cast<ffi.Uint8>().asTypedList(info.byteSize),
        );
      }
    } else if (target is List) {
      if (info.elementCount == 0) {
        return;
      }
      final List<num> view = _numView(
        _outputPointer(index),
        info.type,
        info.elementCount,
        'Output $index',
      );
      // A scalar tensor fills a one-element list.
      final List<int> shape = info.shape.isEmpty ? const <int>[1] : info.shape;
      _fillNested(target, view, shape, 0, 0, info.type, 'Output $index');
    } else {
      throw ArgumentError.value(
        target,
        'target',
        'must be a list, typed data, or a byte buffer',
      );
    }
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

  int get _inputCount => faceBindings.mp_interpreter_input_count(_context);

  List<TfliteTensorInfo> _readInputs() =>
      List<TfliteTensorInfo>.unmodifiable(<TfliteTensorInfo>[
        for (int i = 0; i < _inputCount; i++) _readInfo(i, isInput: true),
      ]);

  int _indexOf(List<TfliteTensorInfo> tensors, String name, String kind) {
    final int index = tensors.indexWhere(
      (TfliteTensorInfo info) => info.name == name,
    );
    if (index < 0) {
      throw ArgumentError.value(name, 'name', 'no $kind tensor has this name');
    }
    return index;
  }

  TfliteTensorInfo _inputInfo(int index) {
    final List<TfliteTensorInfo> tensors = inputs;
    RangeError.checkValidIndex(index, tensors, 'index');
    return tensors[index];
  }

  TfliteTensorInfo _outputInfo(int index) {
    _ensureReady();
    RangeError.checkValidIndex(index, null, 'index', outputCount);
    return _readInfo(index, isInput: false);
  }

  void _writeBytes(int index, TfliteTensorInfo info, Uint8List bytes) {
    if (bytes.length != info.byteSize) {
      throw ArgumentError(
        'Input $index holds ${info.byteSize} bytes, got ${bytes.length}.',
      );
    }
    if (bytes.isEmpty) {
      return;
    }
    _inputPointer(
      index,
    ).cast<ffi.Uint8>().asTypedList(bytes.length).setAll(0, bytes);
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
        quantization: info.quantization_scale == 0
            ? null
            : TfliteQuantization(
                scale: info.quantization_scale,
                zeroPoint: info.quantization_zero_point,
              ),
      );
    } finally {
      pkg_ffi.calloc.free(infoPtr);
    }
  }

  String _lastError(String fallback) =>
      _readCString(faceBindings.mp_interpreter_last_error(_context)) ??
      fallback;

  /// Checks the interpreter is open and reallocates after a resize.
  void _ensureReady() {
    _ensureNotClosed();
    if (_needsAllocation) {
      allocateTensors();
    }
  }

  void _ensureNotClosed() {
    if (_closed) {
      throw StateError('Interpreter already closed.');
    }
  }
}

/// Dimensions of a nested list, read along its first elements.
List<int> _nestedShape(List<Object?> list) {
  final List<int> shape = <int>[];
  Object? level = list;
  while (level is List) {
    shape.add(level.length);
    if (level.isEmpty) {
      break;
    }
    level = level.first;
  }
  return shape;
}

/// Tensor memory as a list of numbers. Bool tensors read and write 0 and 1.
List<num> _numView(
  ffi.Pointer<ffi.Void> pointer,
  TfliteTensorType type,
  int count,
  String what,
) => switch (type) {
  TfliteTensorType.float32 => pointer.cast<ffi.Float>().asTypedList(count),
  TfliteTensorType.float64 => pointer.cast<ffi.Double>().asTypedList(count),
  TfliteTensorType.int8 => pointer.cast<ffi.Int8>().asTypedList(count),
  TfliteTensorType.uint8 ||
  TfliteTensorType.bool => pointer.cast<ffi.Uint8>().asTypedList(count),
  TfliteTensorType.int16 => pointer.cast<ffi.Int16>().asTypedList(count),
  TfliteTensorType.uint16 => pointer.cast<ffi.Uint16>().asTypedList(count),
  TfliteTensorType.int32 => pointer.cast<ffi.Int32>().asTypedList(count),
  TfliteTensorType.uint32 => pointer.cast<ffi.Uint32>().asTypedList(count),
  TfliteTensorType.int64 => pointer.cast<ffi.Int64>().asTypedList(count),
  TfliteTensorType.uint64 => pointer.cast<ffi.Uint64>().asTypedList(count),
  TfliteTensorType.float16 || TfliteTensorType.other => throw UnsupportedError(
    '$what is ${type.name}, which is only available as raw bytes.',
  ),
};

TypedData _copyTyped(List<num> view) => switch (view) {
  Float32List() => Float32List.fromList(view),
  Float64List() => Float64List.fromList(view),
  Int8List() => Int8List.fromList(view),
  Uint8List() => Uint8List.fromList(view),
  Int16List() => Int16List.fromList(view),
  Uint16List() => Uint16List.fromList(view),
  Int32List() => Int32List.fromList(view),
  Uint32List() => Uint32List.fromList(view),
  Int64List() => Int64List.fromList(view),
  Uint64List() => Uint64List.fromList(view),
  _ => throw StateError('Unexpected tensor view ${view.runtimeType}.'),
};

TypedData _emptyTyped(TfliteTensorType type, String what) => switch (type) {
  TfliteTensorType.float32 => Float32List(0),
  TfliteTensorType.float64 => Float64List(0),
  TfliteTensorType.int8 => Int8List(0),
  TfliteTensorType.uint8 || TfliteTensorType.bool => Uint8List(0),
  TfliteTensorType.int16 => Int16List(0),
  TfliteTensorType.uint16 => Uint16List(0),
  TfliteTensorType.int32 => Int32List(0),
  TfliteTensorType.uint32 => Uint32List(0),
  TfliteTensorType.int64 => Int64List(0),
  TfliteTensorType.uint64 => Uint64List(0),
  TfliteTensorType.float16 || TfliteTensorType.other => throw UnsupportedError(
    '$what is ${type.name}, which is only available as raw bytes.',
  ),
};

bool _isTypedListOf(TypedData data, TfliteTensorType type) => switch (type) {
  TfliteTensorType.float32 => data is Float32List,
  TfliteTensorType.float64 => data is Float64List,
  TfliteTensorType.int8 => data is Int8List,
  TfliteTensorType.uint8 || TfliteTensorType.bool => data is Uint8List,
  TfliteTensorType.int16 => data is Int16List,
  TfliteTensorType.uint16 => data is Uint16List,
  TfliteTensorType.int32 => data is Int32List,
  TfliteTensorType.uint32 => data is Uint32List,
  TfliteTensorType.int64 => data is Int64List,
  TfliteTensorType.uint64 => data is Uint64List,
  TfliteTensorType.float16 || TfliteTensorType.other => false,
};

bool _isFloat(TfliteTensorType type) =>
    type == TfliteTensorType.float32 || type == TfliteTensorType.float64;

/// Writes the numbers and bools in [source], depth first, to [view] starting
/// at [offset]. Returns the offset after the last element written.
int _flattenInto(
  List<Object?> source,
  List<num> view,
  int offset,
  TfliteTensorType type,
) {
  final bool isFloat = _isFloat(type);
  for (final Object? element in source) {
    if (element is List) {
      offset = _flattenInto(element, view, offset, type);
      continue;
    }
    if (offset >= view.length) {
      // Keep counting so the caller can report the real element count.
      offset++;
      continue;
    }
    if (element is num) {
      view[offset] = isFloat ? element.toDouble() : element.toInt();
    } else if (element is bool) {
      view[offset] = element ? 1 : 0;
    } else {
      throw ArgumentError(
        'Tensor data must be numbers or bools, got $element.',
      );
    }
    offset++;
  }
  return offset;
}

/// Fills [target], a nested list shaped like [shape], from [view]. Returns
/// the offset after the last element read.
int _fillNested(
  List<Object?> target,
  List<num> view,
  List<int> shape,
  int depth,
  int offset,
  TfliteTensorType type,
  String what,
) {
  if (target.length != shape[depth]) {
    throw ArgumentError(
      '$what has shape $shape, but the target list has length '
      '${target.length} at depth $depth.',
    );
  }
  if (depth < shape.length - 1) {
    for (final Object? child in target) {
      if (child is! List) {
        throw ArgumentError(
          '$what has shape $shape, but the target list is not nested '
          '${shape.length} deep.',
        );
      }
      offset = _fillNested(child, view, shape, depth + 1, offset, type, what);
    }
    return offset;
  }
  final bool isFloat = _isFloat(type);
  final bool isBool = type == TfliteTensorType.bool;
  for (int i = 0; i < target.length; i++) {
    final num value = view[offset++];
    target[i] = isBool
        ? value != 0
        : isFloat
        ? value.toDouble()
        : value.toInt();
  }
  return offset;
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
