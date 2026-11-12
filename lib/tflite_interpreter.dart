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
import 'src/native_helpers.dart';

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

/// Native resources of one interpreter. Kept apart from [TfliteInterpreter]
/// so the finalizer can release them without holding the interpreter.
final class _NativeHandles {
  const _NativeHandles(this.context, this.modelData);

  final ffi.Pointer<MpInterpreterContext> context;

  /// Model bytes the context reads from, or null for a model loaded from a
  /// file. The native side borrows them, so they are freed after the context.
  final ffi.Pointer<ffi.Uint8> modelData;

  void release() {
    faceBindings.mp_interpreter_destroy(context);
    if (modelData != ffi.nullptr) {
      pkg_ffi.malloc.free(modelData);
    }
  }
}

final Finalizer<_NativeHandles> _interpreterFinalizer =
    Finalizer<_NativeHandles>((_NativeHandles handles) => handles.release());

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
  TfliteInterpreter._(this._handles) {
    _interpreterFinalizer.attach(this, _handles, detach: this);
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
  /// The bytes are copied once into native memory, so [model] can be
  /// discarded after this returns.
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
    validateThreads(threads);
    // The native interpreter reads the model from this buffer for as long as
    // it lives. _create takes ownership of it.
    final ffi.Pointer<ffi.Uint8> modelPtr = pkg_ffi.malloc<ffi.Uint8>(
      model.length,
    );
    modelPtr.asTypedList(model.length).setAll(0, model);
    return _create(
      threads: threads,
      delegate: delegate,
      allowDelegateFallback: allowDelegateFallback,
      modelData: modelPtr,
      create: (options) => faceBindings.mp_interpreter_create_from_buffer(
        modelPtr,
        model.length,
        options,
      ),
    );
  }

  /// Loads the model file at [path]. See [TfliteInterpreter.fromBuffer] for
  /// the options.
  ///
  /// Throws [FaceMeshException] when the file cannot be read or is not a
  /// model.
  factory TfliteInterpreter.fromFile(
    String path, {
    int? threads,
    FaceMeshDelegate delegate = FaceMeshDelegate.cpu,
    bool allowDelegateFallback = true,
  }) {
    validateThreads(threads);
    if (Platform.isWindows) {
      // The runtime opens the path with the narrow-character C API, which
      // cannot name a file outside the ANSI code page. Dart can, so read
      // the file here.
      final Uint8List bytes;
      try {
        bytes = File(path).readAsBytesSync();
      } on FileSystemException catch (e) {
        throw FaceMeshException(
          'Unable to load model file: $path (${e.osError?.message ?? e.message})',
        );
      }
      if (bytes.isEmpty) {
        throw FaceMeshException('Unable to load model file: $path');
      }
      return TfliteInterpreter.fromBuffer(
        bytes,
        threads: threads,
        delegate: delegate,
        allowDelegateFallback: allowDelegateFallback,
      );
    }
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
    ffi.Pointer<ffi.Uint8>? modelData,
  }) {
    final optionsPtr = pkg_ffi.calloc<MpInterpreterCreateOptions>();
    try {
      optionsPtr.ref
        ..threads = threads ?? defaultInferenceThreads()
        ..delegate = delegate.index
        ..disable_delegate_fallback = allowDelegateFallback ? 0 : 1;
      final ffi.Pointer<MpInterpreterContext> context = create(optionsPtr);
      if (context == ffi.nullptr) {
        if (modelData != null) {
          pkg_ffi.malloc.free(modelData);
        }
        throw FaceMeshException(
          readCString(faceBindings.mp_interpreter_last_global_error()) ??
              'Failed to create interpreter.',
        );
      }
      // From here the instance owns both, and releases them if its own
      // constructor fails.
      return TfliteInterpreter._(
        _NativeHandles(context, modelData ?? ffi.nullptr),
      );
    } finally {
      pkg_ffi.calloc.free(optionsPtr);
    }
  }

  final _NativeHandles _handles;
  ffi.Pointer<MpInterpreterContext> get _context => _handles.context;
  bool _closed = false;

  /// True between [resizeInput] and the next allocation.
  bool _needsAllocation = false;

  /// Input shapes of the last successful allocation, kept while a resize is
  /// pending so a failed allocation can go back to them.
  List<List<int>>? _shapesBeforeResize;
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
  /// Resize every input that has to change before that allocation.
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
    if (shape.any((int dim) => dim < 0)) {
      throw ArgumentError.value(
        shape,
        'shape',
        'dimensions must not be negative',
      );
    }
    if (!_needsAllocation) {
      _shapesBeforeResize = <List<int>>[
        for (final TfliteTensorInfo info in inputs) info.shape,
      ];
    }
    if (!_resizeNative(index, shape)) {
      throw FaceMeshException(_lastError('Unable to resize input $index.'));
    }
    _needsAllocation = true;
    _inputs = null;
  }

  /// Reallocates tensors after [resizeInput].
  ///
  /// Throws [FaceMeshException] when the model cannot run with the new
  /// shapes. The inputs then go back to the shapes they had before the
  /// resize, so the interpreter stays usable, but the data they held is
  /// gone and has to be set again.
  void allocateTensors() {
    _ensureNotClosed();
    _inputs = null;
    final List<List<int>>? previous = _shapesBeforeResize;
    _shapesBeforeResize = null;
    if (faceBindings.mp_interpreter_allocate_tensors(_context) != 0) {
      _needsAllocation = false;
      return;
    }
    final String message = _lastError('Tensor allocation failed.');
    if (previous != null && _restoreShapes(previous)) {
      throw FaceMeshException(
        '$message The inputs are back to their previous shapes and hold no '
        'data.',
      );
    }
    throw FaceMeshException(message);
  }

  /// Resizes every input to [shapes] and reallocates. Returns false, with
  /// an allocation still pending, when that fails.
  bool _restoreShapes(List<List<int>> shapes) {
    bool restored = true;
    for (int i = 0; i < shapes.length; i++) {
      restored = _resizeNative(i, shapes[i]) && restored;
    }
    restored =
        restored && faceBindings.mp_interpreter_allocate_tensors(_context) != 0;
    _inputs = null;
    _shapesBeforeResize = null;
    _needsAllocation = !restored;
    return restored;
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
  /// [ByteBuffer]). A nested list with another element count resizes that
  /// input to the list's shape, which must have the tensor's rank.
  ///
  /// Each output is a nested list shaped like its tensor, filled in place, or
  /// typed data of the tensor's byte size.
  ///
  /// Throws [ArgumentError] when a value does not fit its tensor. When the
  /// call resized inputs and then fails, they go back to their previous
  /// shapes.
  void runForMultipleInputs(List<Object> inputs, Map<int, Object> outputs) {
    _ensureNotClosed();
    if (inputs.length != _inputCount) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'the model has $_inputCount inputs',
      );
    }
    // Read the shapes once: reading them again after a resize would
    // reallocate before the other inputs are resized.
    final List<TfliteTensorInfo> current = this.inputs;
    // Decide every resize before making any, so a list that is rejected
    // here leaves the interpreter as it was.
    final Map<int, List<int>> resizes = <int, List<int>>{};
    for (int i = 0; i < inputs.length; i++) {
      final Object input = inputs[i];
      if (input is TypedData || input is ByteBuffer || input is! List) {
        continue;
      }
      final List<int> shape = _nestedShape(input);
      final TfliteTensorInfo info = current[i];
      final int count = _countElements(input);
      if (count != shape.fold<int>(1, (int a, int b) => a * b)) {
        throw ArgumentError(
          'Input $i is not a regular nested list: its first elements give '
          'shape $shape, but it holds $count elements.',
        );
      }
      if (count == info.elementCount) {
        continue;
      }
      if (shape.length != info.shape.length) {
        throw ArgumentError(
          'Input $i has shape ${info.shape} (${info.elementCount} elements). '
          'A list of shape $shape does not fit it and cannot resize it: nest '
          'the list ${info.shape.length} deep.',
        );
      }
      resizes[i] = shape;
    }
    if (resizes.isEmpty) {
      _setInputsAndInvoke(inputs);
      outputs.forEach(copyOutputTo);
      return;
    }
    final List<List<int>> previous = <List<int>>[
      for (final TfliteTensorInfo info in current) info.shape,
    ];
    try {
      resizes.forEach(resizeInput);
      _setInputsAndInvoke(inputs);
      outputs.forEach(copyOutputTo);
    } catch (_) {
      _restoreShapes(previous);
      rethrow;
    }
  }

  void _setInputsAndInvoke(List<Object> inputs) {
    for (int i = 0; i < inputs.length; i++) {
      setInput(i, inputs[i]);
    }
    invoke();
  }

  /// Writes [data] to the input at [index].
  ///
  /// [data] is a typed list of the tensor's type (a [Float32List] for a
  /// float32 tensor), raw bytes ([Uint8List], [ByteData], [ByteBuffer]), or a
  /// flat or nested list of numbers or bools in row-major order. A single
  /// number or bool sets a one-element tensor.
  ///
  /// Throws [ArgumentError] when [data] does not match the tensor's type or
  /// size.
  void setInput(int index, Object data) {
    if (data is num || data is bool) {
      setInput(index, <Object>[data]);
      return;
    }
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
      // Convert into a Dart buffer first, so a list that turns out not to
      // fit leaves the tensor untouched.
      final List<num> staged =
          _newTyped(info.type, info.elementCount, 'Input $index') as List<num>;
      final int count = _flattenInto(
        data,
        staged,
        0,
        info.type,
        'Input $index',
      );
      if (count != info.elementCount) {
        throw ArgumentError(
          'Input $index holds ${info.elementCount} elements, got $count.',
        );
      }
      if (count > 0) {
        _numView(
          _inputPointer(index),
          info.type,
          count,
          'Input $index',
        ).setAll(0, staged);
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
    return _copyOutput(index, info) as Float32List;
  }

  /// Returns a copy of the output at [index] as a typed list of its type: a
  /// [Float32List] for float32, an [Int32List] for int32, and so on. Bool
  /// tensors come back as a [Uint8List] of 0 and 1.
  ///
  /// Throws [UnsupportedError] for float16 and [TfliteTensorType.other]
  /// tensors. Use [outputBytes] for those.
  TypedData outputData(int index) => _copyOutput(index, _outputInfo(index));

  TypedData _copyOutput(int index, TfliteTensorInfo info) {
    final int count = info.elementCount;
    if (count == 0) {
      return _newTyped(info.type, 0, 'Output $index');
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
  /// are filled in place, a typed list of the tensor's type, or raw bytes
  /// ([Uint8List], [ByteData], [ByteBuffer]) of the tensor's byte size.
  ///
  /// Throws [ArgumentError] when [target] does not fit the tensor.
  void copyOutputTo(int index, Object target) {
    final TfliteTensorInfo info = _outputInfo(index);
    if (target is TypedData &&
        target is! Uint8List &&
        target is! ByteData &&
        !_isTypedListOf(target, info.type)) {
      throw ArgumentError(
        'Output $index is ${info.type.name}, but the target is a '
        '${target.runtimeType}.',
      );
    }
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
      // An empty tensor has no memory to view, but the target still has to
      // match its shape, so stale values are not mistaken for a result.
      final List<num> view = info.elementCount == 0
          ? const <num>[]
          : _numView(
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
    _handles.release();
    _closed = true;
  }

  bool _resizeNative(int index, List<int> shape) {
    final ffi.Pointer<ffi.Int32> dimsPtr = pkg_ffi.calloc<ffi.Int32>(
      shape.isEmpty ? 1 : shape.length,
    );
    try {
      dimsPtr.asTypedList(shape.length).setAll(0, shape);
      return faceBindings.mp_interpreter_resize_input(
            _context,
            index,
            dimsPtr,
            shape.length,
          ) !=
          0;
    } finally {
      pkg_ffi.calloc.free(dimsPtr);
    }
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
        name: readCString(info.name) ?? '',
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
      readCString(faceBindings.mp_interpreter_last_error(_context)) ?? fallback;

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

/// A zeroed typed list of [count] elements of [type].
TypedData _newTyped(TfliteTensorType type, int count, String what) =>
    switch (type) {
      TfliteTensorType.float32 => Float32List(count),
      TfliteTensorType.float64 => Float64List(count),
      TfliteTensorType.int8 => Int8List(count),
      TfliteTensorType.uint8 || TfliteTensorType.bool => Uint8List(count),
      TfliteTensorType.int16 => Int16List(count),
      TfliteTensorType.uint16 => Uint16List(count),
      TfliteTensorType.int32 => Int32List(count),
      TfliteTensorType.uint32 => Uint32List(count),
      TfliteTensorType.int64 => Int64List(count),
      TfliteTensorType.uint64 => Uint64List(count),
      TfliteTensorType.float16 ||
      TfliteTensorType.other => throw UnsupportedError(
        '$what is ${type.name}, which is only available as raw bytes.',
      ),
    };

bool _isTypedListOf(TypedData data, TfliteTensorType type) => switch (type) {
  TfliteTensorType.float32 => data is Float32List,
  TfliteTensorType.float64 => data is Float64List,
  TfliteTensorType.int8 => data is Int8List,
  TfliteTensorType.uint8 ||
  TfliteTensorType.bool => data is Uint8List || data is Uint8ClampedList,
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
  String what,
) {
  final bool isFloat = _isFloat(type);
  final (int, int)? range = _intRange(type);
  for (final Object? element in source) {
    if (element is List) {
      offset = _flattenInto(element, view, offset, type, what);
      continue;
    }
    if (offset >= view.length) {
      // Keep counting so the caller can report the real element count.
      offset++;
      continue;
    }
    if (element is num) {
      if (isFloat) {
        view[offset] = element.toDouble();
      } else if (element is double &&
          (!element.isFinite || element != element.truncateToDouble())) {
        throw ArgumentError(
          '$what is ${type.name}, so $element is not a value it can hold.',
        );
      } else {
        final int value = element.toInt();
        if (range != null && (value < range.$1 || value > range.$2)) {
          throw ArgumentError(
            '$what is ${type.name}, which holds ${range.$1} to ${range.$2}. '
            'Got $value.',
          );
        }
        view[offset] = value;
      }
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

/// Smallest and largest value of an integer tensor type, or null for types
/// a Dart int always fits or that are not integers.
(int, int)? _intRange(TfliteTensorType type) => switch (type) {
  TfliteTensorType.bool => (0, 1),
  TfliteTensorType.int8 => (-128, 127),
  TfliteTensorType.uint8 => (0, 255),
  TfliteTensorType.int16 => (-32768, 32767),
  TfliteTensorType.uint16 => (0, 65535),
  TfliteTensorType.int32 => (-2147483648, 2147483647),
  TfliteTensorType.uint32 => (0, 4294967295),
  _ => null,
};

/// Number of numbers and bools in [list], however it is nested.
int _countElements(List<Object?> list) {
  int count = 0;
  for (final Object? element in list) {
    count += element is List ? _countElements(element) : 1;
  }
  return count;
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
  try {
    for (int i = 0; i < target.length; i++) {
      final num value = view[offset++];
      target[i] = isBool
          ? value != 0
          : isFloat
          ? value.toDouble()
          : value.toInt();
    }
    // A list of another element type rejects the assignment.
    // ignore: avoid_catching_errors
  } on TypeError {
    final String element = isBool
        ? 'bool'
        : isFloat
        ? 'double'
        : 'int';
    throw ArgumentError(
      '$what is ${type.name}, so its innermost target lists must hold '
      '$element values, for example List<$element>.filled(n, ...). Got a '
      '${target.runtimeType}.',
    );
    // ignore: avoid_catching_errors
  } on UnsupportedError {
    throw ArgumentError(
      '$what cannot be copied into an unmodifiable list. Pass lists that '
      'can be written to.',
    );
  }
  return offset;
}
