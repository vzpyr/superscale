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
        job.completer.completeError(StateError(list[2] as String));
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

void upscaleWorkerMain(SendPort ready) {
  final inbox = ReceivePort();
  ready.send(inbox.sendPort);
  final sessions = <String, OrtSession>{};
  var cancelled = false;

  inbox.listen((message) {
    final list = message as List<Object?>;
    if (list.first == 'cancel') {
      cancelled = true;
      return;
    }
    cancelled = false;
    unawaited(_handle(list, sessions, () => cancelled));
  });
}

Future<void> _handle(
  List<Object?> list,
  Map<String, OrtSession> sessions,
  bool Function() isCancelled,
) async {
  final id = list[0] as int;
  final reply = list[1] as SendPort;
  try {
    final output = await _upscale(
      sessions: sessions,
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
    reply.send([id, 'error', error.toString()]);
  }
}

Future<UpscaleOutput> _upscale({
  required Map<String, OrtSession> sessions,
  required String modelPath,
  required int threads,
  required Uint8List pixels,
  required int width,
  required int height,
  required bool Function() isCancelled,
  required void Function(int done, int total) onProgress,
}) async {
  var session = sessions[modelPath];
  if (session == null) {
    session = _openSession(modelPath, threads);
    sessions[modelPath] = session;
  }

  final scale = session.scale;
  final tile = _pickTile(width, height);
  final margin = tile >= 128 ? 16 : 8;
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

        final tilePixels = result.asTypedList(
          produced.width * produced.height * 4,
        );
        _blit(
          tilePixels,
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

OrtSession _openSession(String modelPath, int threads) {
  return OrtEngine.open(modelPath: modelPath, threads: threads);
}

int _pickTile(int width, int height) {
  final longest = width > height ? width : height;
  if (longest <= 64) return 64;
  if (longest <= 128) return 128;
  return 256;
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
  final startX = coreX - margin;
  final startY = coreY - margin;
  final stride = width * 4;

  if (startX >= 0 &&
      startY >= 0 &&
      startX + span <= width &&
      startY + span <= height) {
    for (var y = 0; y < span; y++) {
      final sourceOffset = (startY + y) * stride + startX * 4;
      view.setRange(y * span * 4, (y + 1) * span * 4, source, sourceOffset);
    }
    return;
  }

  var offset = 0;
  for (var y = 0; y < span; y++) {
    var sourceY = startY + y;
    if (sourceY < 0) {
      sourceY = 0;
    } else if (sourceY >= height) {
      sourceY = height - 1;
    }
    final row = sourceY * stride;
    for (var x = 0; x < span; x++) {
      var sourceX = startX + x;
      if (sourceX < 0) {
        sourceX = 0;
      } else if (sourceX >= width) {
        sourceX = width - 1;
      }
      final index = row + sourceX * 4;
      view[offset] = source[index];
      view[offset + 1] = source[index + 1];
      view[offset + 2] = source[index + 2];
      view[offset + 3] = 255;
      offset += 4;
    }
  }
}

void _blit(
  Uint8List tile,
  int tileWidth,
  int destWidth,
  int destX,
  int destY,
  int copyWidth,
  int copyHeight,
  int margin,
  Uint8List target,
) {
  final sourceStride = tileWidth * 4;
  final targetStride = destWidth * 4;
  final rowBytes = copyWidth * 4;
  for (var y = 0; y < copyHeight; y++) {
    final sourceOffset = (margin + y) * sourceStride + margin * 4;
    final targetOffset = (destY + y) * targetStride + destX * 4;
    target.setRange(targetOffset, targetOffset + rowBytes, tile, sourceOffset);
  }
}
