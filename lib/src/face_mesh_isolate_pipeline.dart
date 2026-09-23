part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Builds the [FaceMeshInferencePipeline] that a [FaceMeshIsolatePipeline]
/// runs. Called once inside the worker isolate with the argument given to
/// [FaceMeshIsolatePipeline.spawn].
///
/// Use a top-level or static function, as with `Isolate.spawn`. A closure
/// made inside a `State` captures `this`, which cannot be sent.
typedef FaceMeshPipelineFactory<T> =
    FutureOr<FaceMeshInferencePipeline> Function(T argument);

/// Runs a [FaceMeshInferencePipeline] in a worker isolate and exposes its
/// methods as `Future`s. The calling isolate pays only for copying the frame
/// in and the result out.
///
/// ```dart
/// Future<FaceMeshInferencePipeline> createPipeline(FaceMeshModel model) async =>
///     FaceMeshInferencePipeline(
///       detector: await FaceDetectorProcessor.create(),
///       mesh: await FaceMeshProcessor.create(model: model),
///     );
///
/// final FaceMeshIsolatePipeline pipeline = await FaceMeshIsolatePipeline.spawn(
///   createPipeline,
///   FaceMeshModel.v2,
/// );
/// final FaceMeshInferenceResult result = await pipeline.process(frame);
/// ```
///
/// Requests run one at a time in the order they are sent. Tracking and
/// smoothing state stays in the worker across calls. The frame's pixel
/// buffer is copied into the worker on every call, and the result is copied
/// back. [close] closes the pipeline in the worker and shuts the worker
/// down.
class FaceMeshIsolatePipeline {
  FaceMeshIsolatePipeline._(this._fromWorker);

  final ReceivePort _fromWorker;

  /// The worker's request port, set once the pipeline is built there.
  SendPort? _toWorker;
  final Map<int, Completer<Object?>> _pending = <int, Completer<Object?>>{};
  int _nextRequestId = 0;
  bool _closed = false;

  /// Spawns the worker isolate and builds the pipeline in it by calling
  /// [createPipeline] with [argument].
  ///
  /// [argument] is copied to the worker, so it must be sendable. Enums,
  /// numbers, strings, records, or plain data classes of those work. Pass
  /// `null` when the factory needs nothing. Must be called from the root
  /// isolate, so the worker can load the bundled model assets. Throws
  /// whatever [createPipeline] throws.
  ///
  /// The worker opens the native library on its own. A path given to
  /// `initializeFaceBindings(libraryPath:)` carries over, a `dylib` handle
  /// does not.
  static Future<FaceMeshIsolatePipeline> spawn<T>(
    FaceMeshPipelineFactory<T> createPipeline,
    T argument, {
    String debugName = 'FaceMeshIsolatePipeline',
  }) async {
    final ReceivePort fromWorker = ReceivePort();
    final FaceMeshIsolatePipeline pipeline = FaceMeshIsolatePipeline._(
      fromWorker,
    );
    final Completer<Object?> ready = pipeline._register(_IsolateRequest.bootId);
    fromWorker.listen(pipeline._onWorkerMessage);
    try {
      try {
        await Isolate.spawn<_IsolateBootstrap>(
          _faceMeshIsolateMain,
          _IsolateBootstrap(
            reply: fromWorker.sendPort,
            libraryPath: faceLibraryPath,
            createPipeline: createPipeline,
            argument: argument,
          ),
          debugName: debugName,
          onExit: fromWorker.sendPort,
        );
      } on ArgumentError catch (error) {
        // The VM rejected the message. A closure or instance method carries
        // its scope, and an argument may hold something unsendable.
        throw ArgumentError(
          'createPipeline must be a top-level or static function, and '
          'argument must be sendable to an isolate. ${error.message}',
        );
      }
      pipeline._toWorker = await ready.future as SendPort;
      return pipeline;
    } catch (_) {
      pipeline._closed = true;
      fromWorker.close();
      rethrow;
    }
  }

  /// Whether the pipeline is closed, by [close] or because the worker
  /// exited.
  bool get isClosed => _closed;

  /// Whether a request is waiting for the worker.
  ///
  /// Requests queue in the worker, so a camera loop should skip the frame
  /// while this is true instead of sending another.
  bool get isBusy => _pending.isNotEmpty;

  /// Runs [FaceMeshInferencePipeline.process] in the worker.
  Future<FaceMeshInferenceResult> process(
    FaceMeshFrame frame, {
    NormalizedRect? detectorRoi,
    bool runMesh = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
    Duration? timestamp,
  }) async {
    final Object? result = await _send(
      _IsolateRequestKind.process,
      frame: frame,
      detectorRoi: detectorRoi,
      runMesh: runMesh,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      detectorRoiScaleX: detectorRoiScaleX,
      detectorRoiScaleY: detectorRoiScaleY,
      detectorRoiShiftX: detectorRoiShiftX,
      detectorRoiShiftY: detectorRoiShiftY,
      timestamp: timestamp,
    );
    return result as FaceMeshInferenceResult;
  }

  /// Runs [FaceMeshInferencePipeline.processMultiFace] in the worker.
  Future<FaceMeshMultiInferenceResult> processMultiFace(
    FaceMeshFrame frame, {
    required int maxMeshFaces,
    NormalizedRect? detectorRoi,
    bool runMesh = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
    Duration? timestamp,
  }) async {
    final Object? result = await _send(
      _IsolateRequestKind.processMultiFace,
      frame: frame,
      maxMeshFaces: maxMeshFaces,
      detectorRoi: detectorRoi,
      runMesh: runMesh,
      rotationDegrees: rotationDegrees,
      mirrorHorizontal: mirrorHorizontal,
      detectorRoiScaleX: detectorRoiScaleX,
      detectorRoiScaleY: detectorRoiScaleY,
      detectorRoiShiftX: detectorRoiShiftX,
      detectorRoiShiftY: detectorRoiShiftY,
      timestamp: timestamp,
    );
    return result as FaceMeshMultiInferenceResult;
  }

  /// Runs [FaceMeshInferencePipeline.resetTracking] in the worker.
  Future<void> resetTracking() async {
    await _send(_IsolateRequestKind.resetTracking);
  }

  /// Calls [FaceMeshInferencePipeline.close] in the worker and shuts the
  /// worker down.
  ///
  /// Requests already sent complete first. Calling this more than once is a
  /// no-op.
  Future<void> close() async {
    if (_closed) {
      return;
    }
    _closed = true;
    try {
      await _send(_IsolateRequestKind.close, checkClosed: false);
    } finally {
      _failPending(FaceMeshException('FaceMeshIsolatePipeline closed.'));
      _fromWorker.close();
    }
  }

  Future<Object?> _send(
    _IsolateRequestKind kind, {
    bool checkClosed = true,
    FaceMeshFrame? frame,
    int? maxMeshFaces,
    NormalizedRect? detectorRoi,
    bool runMesh = true,
    int rotationDegrees = 0,
    bool mirrorHorizontal = false,
    double? detectorRoiScaleX,
    double? detectorRoiScaleY,
    double? detectorRoiShiftX,
    double? detectorRoiShiftY,
    Duration? timestamp,
  }) {
    if (checkClosed && _closed) {
      throw StateError('FaceMeshIsolatePipeline is closed.');
    }
    final SendPort? toWorker = _toWorker;
    if (toWorker == null) {
      throw StateError('FaceMeshIsolatePipeline worker is not running.');
    }
    final int id = _nextRequestId++;
    final Completer<Object?> completer = _register(id);
    try {
      toWorker.send(
        _IsolateRequest(
          id: id,
          kind: kind,
          frame: frame,
          maxMeshFaces: maxMeshFaces,
          detectorRoi: detectorRoi,
          runMesh: runMesh,
          rotationDegrees: rotationDegrees,
          mirrorHorizontal: mirrorHorizontal,
          detectorRoiScaleX: detectorRoiScaleX,
          detectorRoiScaleY: detectorRoiScaleY,
          detectorRoiShiftX: detectorRoiShiftX,
          detectorRoiShiftY: detectorRoiShiftY,
          timestamp: timestamp,
        ),
      );
    } catch (_) {
      _pending.remove(id);
      rethrow;
    }
    return completer.future;
  }

  Completer<Object?> _register(int id) {
    final Completer<Object?> completer = Completer<Object?>();
    _pending[id] = completer;
    return completer;
  }

  void _onWorkerMessage(Object? message) {
    if (message is _AssetRequest) {
      _serveAsset(message);
      return;
    }
    if (message is _IsolateReply) {
      final Completer<Object?>? completer = _pending.remove(message.id);
      if (completer == null) {
        return;
      }
      final Object? error = message.error;
      if (error != null) {
        completer.completeError(
          error,
          StackTrace.fromString(message.stackTrace ?? ''),
        );
      } else {
        completer.complete(message.value);
      }
      return;
    }
    // A null message is the onExit notification. It arrives after the close
    // reply in the normal case, and without one if the worker died.
    if (message == null) {
      _closed = true;
      _failPending(FaceMeshException('FaceMeshIsolatePipeline worker exited.'));
      _fromWorker.close();
    }
  }

  /// Loads a model asset on behalf of the worker, which has no
  /// `ServicesBinding` and so cannot use `rootBundle` itself.
  void _serveAsset(_AssetRequest request) {
    // Future.sync turns a synchronous throw from load into a reply.
    Future<ByteData>.sync(() => rootBundle.load(request.key)).then(
      (ByteData data) => request.reply.send(_AssetReply(data: data)),
      onError: (Object error) =>
          request.reply.send(_AssetReply(error: error.toString())),
    );
  }

  void _failPending(Object error) {
    final List<Completer<Object?>> pending = _pending.values.toList();
    _pending.clear();
    for (final Completer<Object?> completer in pending) {
      completer.completeError(error, StackTrace.current);
    }
  }
}

enum _IsolateRequestKind { process, processMultiFace, resetTracking, close }

class _IsolateBootstrap {
  const _IsolateBootstrap({
    required this.reply,
    required this.libraryPath,
    required this.createPipeline,
    required this.argument,
  });

  final SendPort reply;
  final String? libraryPath;

  /// Typed as [Function] so the bootstrap and the isolate entry point stay
  /// non-generic. The worker calls it with [argument].
  final Function createPipeline;
  final Object? argument;
}

class _IsolateRequest {
  const _IsolateRequest({
    required this.id,
    required this.kind,
    this.frame,
    this.maxMeshFaces,
    this.detectorRoi,
    this.runMesh = true,
    this.rotationDegrees = 0,
    this.mirrorHorizontal = false,
    this.detectorRoiScaleX,
    this.detectorRoiScaleY,
    this.detectorRoiShiftX,
    this.detectorRoiShiftY,
    this.timestamp,
  });

  /// Id of the reply that carries the worker's request port.
  static const int bootId = -1;

  final int id;
  final _IsolateRequestKind kind;
  final FaceMeshFrame? frame;
  final int? maxMeshFaces;
  final NormalizedRect? detectorRoi;
  final bool runMesh;
  final int rotationDegrees;
  final bool mirrorHorizontal;
  final double? detectorRoiScaleX;
  final double? detectorRoiScaleY;
  final double? detectorRoiShiftX;
  final double? detectorRoiShiftY;
  final Duration? timestamp;
}

class _IsolateReply {
  const _IsolateReply(this.id, {this.value, this.error, this.stackTrace});

  final int id;
  final Object? value;
  final Object? error;
  final String? stackTrace;
}

Future<void> _faceMeshIsolateMain(_IsolateBootstrap bootstrap) async {
  final SendPort reply = bootstrap.reply;
  final FaceMeshInferencePipeline pipeline;
  try {
    _assetRelayPort = reply;
    final String? libraryPath = bootstrap.libraryPath;
    if (libraryPath != null) {
      initializeFaceBindings(libraryPath: libraryPath);
    }
    final Object? built = await bootstrap.createPipeline(bootstrap.argument);
    pipeline = built as FaceMeshInferencePipeline;
  } catch (error, stackTrace) {
    _replyTo(
      reply,
      _IsolateRequest.bootId,
      error: error,
      stackTrace: stackTrace,
    );
    Isolate.exit();
  }

  final ReceivePort requests = ReceivePort();
  reply.send(_IsolateReply(_IsolateRequest.bootId, value: requests.sendPort));

  await for (final Object? message in requests) {
    final _IsolateRequest request = message as _IsolateRequest;
    Object? value;
    try {
      value = switch (request.kind) {
        _IsolateRequestKind.process => pipeline.process(
          request.frame!,
          detectorRoi: request.detectorRoi,
          runMesh: request.runMesh,
          rotationDegrees: request.rotationDegrees,
          mirrorHorizontal: request.mirrorHorizontal,
          detectorRoiScaleX: request.detectorRoiScaleX,
          detectorRoiScaleY: request.detectorRoiScaleY,
          detectorRoiShiftX: request.detectorRoiShiftX,
          detectorRoiShiftY: request.detectorRoiShiftY,
          timestamp: request.timestamp,
        ),
        _IsolateRequestKind.processMultiFace => pipeline.processMultiFace(
          request.frame!,
          maxMeshFaces: request.maxMeshFaces!,
          detectorRoi: request.detectorRoi,
          runMesh: request.runMesh,
          rotationDegrees: request.rotationDegrees,
          mirrorHorizontal: request.mirrorHorizontal,
          detectorRoiScaleX: request.detectorRoiScaleX,
          detectorRoiScaleY: request.detectorRoiScaleY,
          detectorRoiShiftX: request.detectorRoiShiftX,
          detectorRoiShiftY: request.detectorRoiShiftY,
          timestamp: request.timestamp,
        ),
        _IsolateRequestKind.resetTracking => _run(pipeline.resetTracking),
        _IsolateRequestKind.close => _run(pipeline.close),
      };
    } catch (error, stackTrace) {
      _replyTo(reply, request.id, error: error, stackTrace: stackTrace);
      if (request.kind == _IsolateRequestKind.close) {
        requests.close();
      }
      continue;
    }
    _replyTo(reply, request.id, value: value);
    if (request.kind == _IsolateRequestKind.close) {
      requests.close();
    }
  }
  // Ends the isolate even if the factory left a port open, so onExit fires.
  Isolate.exit();
}

Object? _run(void Function() action) {
  action();
  return null;
}

void _replyTo(
  SendPort reply,
  int id, {
  Object? value,
  Object? error,
  StackTrace? stackTrace,
}) {
  final String? trace = stackTrace?.toString();
  try {
    reply.send(
      _IsolateReply(id, value: value, error: error, stackTrace: trace),
    );
  } on ArgumentError {
    // The value or error object could not be sent. Forward its text instead.
    reply.send(
      _IsolateReply(
        id,
        error: FaceMeshException(
          'Could not send reply to the caller: ${error ?? value}',
        ),
        stackTrace: trace,
      ),
    );
  }
}
