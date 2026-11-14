import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'cancel.dart';
import 'native.dart';

class UpscaleOutput {
  const UpscaleOutput({
    required this.pixels,
    required this.width,
    required this.height,
  });

  final Uint8List pixels;
  final int width;
  final int height;
}

class UpscaleFailure implements Exception {
  const UpscaleFailure(this.reason);

  final String reason;

  @override
  String toString() => reason;
}

const int maxOutputSide = 8192;
const int maxOutputPixels = 24000000;

class UpscaleWorker {
  UpscaleWorker._(this._commands, this._events);

  final SendPort _commands;
  final ReceivePort _events;
  final Map<int, _Job> _jobs = {};
  int _nextId = 0;

  static UpscaleWorker? _instance;
  static Future<UpscaleWorker>? _spawning;

  static Future<UpscaleWorker> instance() {
    final existing = _instance;
    if (existing != null) return Future.value(existing);
    return _spawning ??= _spawn().then((worker) {
      _instance = worker;
      _spawning = null;
      return worker;
    });
  }

  static Future<UpscaleWorker> _spawn() async {
    final events = ReceivePort();
    final ready = Completer<SendPort>();
    late final UpscaleWorker worker;
    events.listen((message) {
      if (message is SendPort) {
        ready.complete(message);
      } else {
        worker._onEvent(message);
      }
    });
    await Isolate.spawn(upscaleWorkerMain, events.sendPort);
    worker = UpscaleWorker._(await ready.future, events);
    return worker;
  }

  Future<UpscaleOutput> upscale({
    required Backend backend,
    required String modelPath,
    required int threads,
    required Uint8List pixels,
    required int width,
    required int height,
    required void Function(int done, int total) onProgress,
  }) {
    final id = _nextId++;
    final job = _Job(onProgress);
    _jobs[id] = job;
    _commands.send([
      id,
      _events.sendPort,
      TransferableTypedData.fromList([pixels]),
      modelPath,
      threads,
      width,
      height,
      backend.index,
    ]);
    return job.completer.future;
  }

  void _onEvent(Object? message) {
    final list = message as List<Object?>;
    final job = _jobs[list[0] as int];
    if (job == null) return;
    switch (list[1]) {
      case 'progress':
        job.onProgress(list[2] as int, list[3] as int);
      case 'done':
        _jobs.remove(list[0] as int);
        job.completer.complete(
          UpscaleOutput(
            pixels: (list[2] as TransferableTypedData)
                .materialize()
                .asUint8List(),
            width: list[3] as int,
            height: list[4] as int,
          ),
        );
      case 'cancelled':
        _jobs.remove(list[0] as int);
        job.completer.completeError(const UpscaleCancelled());
      case 'error':
        _jobs.remove(list[0] as int);
        job.completer.completeError(UpscaleFailure(list[2] as String));
    }
  }

  void cancel() {
    _commands.send(const ['cancel']);
  }
}

class _Job {
  _Job(this.onProgress);

  final void Function(int done, int total) onProgress;
  final Completer<UpscaleOutput> completer = Completer();
}

class _SessionKey {
  const _SessionKey(this.backend, this.modelPath);

  final Backend backend;
  final String modelPath;

  @override
  bool operator ==(Object other) =>
      other is _SessionKey &&
      other.backend == backend &&
      other.modelPath == modelPath;

  @override
  int get hashCode => Object.hash(backend, modelPath);
}

void upscaleWorkerMain(SendPort ready) {
  final inbox = ReceivePort();
  ready.send(inbox.sendPort);
  final sessions = <_SessionKey, Session>{};
  var cancelled = false;
  var queue = Future<void>.value();

  inbox.listen((message) {
    final list = message as List<Object?>;
    if (list.first == 'cancel') {
      cancelled = true;
      return;
    }
    cancelled = false;
    queue = queue.then((_) => _handle(list, sessions, () => cancelled));
  });
}

Future<void> _handle(
  List<Object?> list,
  Map<_SessionKey, Session> sessions,
  bool Function() isCancelled,
) async {
  final id = list[0] as int;
  final reply = list[1] as SendPort;
  try {
    final output = await _upscale(
      sessions: sessions,
      backend: Backend.values[list[7] as int],
      modelPath: list[3] as String,
      threads: list[4] as int,
      pixels: (list[2] as TransferableTypedData).materialize().asUint8List(),
      width: list[5] as int,
      height: list[6] as int,
      isCancelled: isCancelled,
      onProgress: (done, total) => reply.send([id, 'progress', done, total]),
    );
    reply.send([
      id,
      'done',
      TransferableTypedData.fromList([output.pixels]),
      output.width,
      output.height,
    ]);
  } on UpscaleCancelled {
    reply.send([id, 'cancelled']);
  } catch (error) {
    reply.send([id, 'error', _reason(error)]);
  }
}

Session _session(
  Map<_SessionKey, Session> sessions,
  Backend backend,
  String modelPath,
  int threads,
) {
  final key = _SessionKey(backend, modelPath);
  final existing = sessions[key];
  if (existing != null) return existing;
  for (final session in sessions.values) {
    session.close();
  }
  sessions.clear();
  final opened = Session.open(
    modelPath: modelPath,
    backend: backend,
    threads: threads,
  );
  sessions[key] = opened;
  return opened;
}

({int tile, int margin}) _tiling(
  Backend backend,
  int width,
  int height,
  int maxSpan,
) {
  final longest = width > height ? width : height;
  if (backend == Backend.cpu) {
    if (longest <= 64) return (tile: 64, margin: 8);
    if (longest <= 128) return (tile: 128, margin: 16);
    return (tile: 256, margin: 16);
  }
  final limit = maxSpan > 0 ? maxSpan : 64;
  ({int tile, int margin}) best = (tile: limit, margin: 0);
  for (final tile in const [64, 96, 128, 192, 256]) {
    if (tile > limit) break;
    final margin = (limit - tile) ~/ 2 < tile ~/ 8
        ? (limit - tile) ~/ 2
        : tile ~/ 8;
    best = (tile: tile, margin: margin);
    if (tile >= longest) break;
  }
  return best;
}

Future<UpscaleOutput> _upscale({
  required Map<_SessionKey, Session> sessions,
  required Backend backend,
  required String modelPath,
  required int threads,
  required Uint8List pixels,
  required int width,
  required int height,
  required bool Function() isCancelled,
  required void Function(int done, int total) onProgress,
}) async {
  final session = _session(sessions, backend, modelPath, threads);
  final scale = session.scale;
  if (width * scale > maxOutputSide ||
      height * scale > maxOutputSide ||
      width * scale * height * scale > maxOutputPixels) {
    throw const UpscaleFailure('output exceeds 8192 px or 24 MP');
  }
  final tiling = _tiling(backend, width, height, session.maxSpan);
  final tile = tiling.tile;
  final margin = tiling.margin;
  final span = tile + margin * 2;
  final tilesX = (width + tile - 1) ~/ tile;
  final tilesY = (height + tile - 1) ~/ tile;
  final total = tilesX * tilesY;

  final outWidth = width * scale;
  final outHeight = height * scale;
  final output = Uint8List(outWidth * outHeight * 4);

  final input = malloc<Uint8>(span * span * 4);
  final result = malloc<Uint8>(span * scale * span * scale * 4);
  try {
    var done = 0;
    for (var ty = 0; ty < tilesY; ty++) {
      for (var tx = 0; tx < tilesX; tx++) {
        if (isCancelled()) throw const UpscaleCancelled();
        final coreX = tx * tile;
        final coreY = ty * tile;
        final coreWidth = width - coreX < tile ? width - coreX : tile;
        final coreHeight = height - coreY < tile ? height - coreY : tile;

        _fillTile(pixels, width, height, coreX, coreY, span, margin, input);

        final produced = session.run(
          rgba: input,
          width: span,
          height: span,
          out: result,
          outCapacity: span * scale * span * scale * 4,
        );
        if (produced.width != span * scale || produced.height != span * scale) {
          throw const UpscaleFailure('unexpected output size');
        }

        _blit(
          result.asTypedList(produced.width * produced.height * 4),
          produced.width,
          outWidth,
          coreX * scale,
          coreY * scale,
          coreWidth * scale,
          coreHeight * scale,
          margin * scale,
          output,
        );

        done++;
        onProgress(done, total);
        await Future<void>.delayed(Duration.zero);
      }
    }
  } finally {
    malloc.free(input);
    malloc.free(result);
  }

  return UpscaleOutput(pixels: output, width: outWidth, height: outHeight);
}

void _fillTile(
  Uint8List source,
  int width,
  int height,
  int coreX,
  int coreY,
  int span,
  int margin,
  Pointer<Uint8> target,
) {
  final view = target.asTypedList(span * span * 4);
  for (var y = 0; y < span; y++) {
    final srcY = (coreY - margin + y).clamp(0, height - 1);
    for (var x = 0; x < span; x++) {
      final srcX = (coreX - margin + x).clamp(0, width - 1);
      final from = (srcY * width + srcX) * 4;
      final to = (y * span + x) * 4;
      view[to] = source[from];
      view[to + 1] = source[from + 1];
      view[to + 2] = source[from + 2];
      view[to + 3] = 255;
    }
  }
}

void _blit(
  Uint8List tile,
  int tileWidth,
  int outWidth,
  int dstX,
  int dstY,
  int copyWidth,
  int copyHeight,
  int offset,
  Uint8List output,
) {
  for (var y = 0; y < copyHeight; y++) {
    final srcRow = ((y + offset) * tileWidth + offset) * 4;
    final dstRow = ((dstY + y) * outWidth + dstX) * 4;
    output.setRange(dstRow, dstRow + copyWidth * 4, tile, srcRow);
  }
}

String _reason(Object error) => error is UpscaleFailure
    ? error.reason
    : error is EngineFailure
    ? error.message
    : 'unknown error';
