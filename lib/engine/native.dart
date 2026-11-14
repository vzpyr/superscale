import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

enum Backend { cpu, gpu }

class EngineFailure implements Exception {
  const EngineFailure(this.message);

  final String message;

  @override
  String toString() => message;
}

typedef _OpenNative = Pointer<Void> Function(
  Pointer<Utf8> path,
  Int32 backend,
  Int32 threads,
  Pointer<Int32> scale,
  Pointer<Int32> maxSpan,
  Pointer<Pointer<Uint8>> err,
);
typedef _Open = Pointer<Void> Function(
  Pointer<Utf8> path,
  int backend,
  int threads,
  Pointer<Int32> scale,
  Pointer<Int32> maxSpan,
  Pointer<Pointer<Uint8>> err,
);

typedef _RunNative = Int32 Function(
  Pointer<Void> engine,
  Pointer<Uint8> rgba,
  Int32 width,
  Int32 height,
  Pointer<Uint8> out,
  IntPtr outLen,
  Pointer<Int32> outWidth,
  Pointer<Int32> outHeight,
  Pointer<Pointer<Uint8>> err,
);
typedef _Run = int Function(
  Pointer<Void> engine,
  Pointer<Uint8> rgba,
  int width,
  int height,
  Pointer<Uint8> out,
  int outLen,
  Pointer<Int32> outWidth,
  Pointer<Int32> outHeight,
  Pointer<Pointer<Uint8>> err,
);

typedef _CloseNative = Void Function(Pointer<Void> engine);
typedef _Close = void Function(Pointer<Void> engine);
typedef _FreeNative = Void Function(Pointer<Uint8> text);
typedef _Free = void Function(Pointer<Uint8> text);

class Engine {
  static final DynamicLibrary _lib = DynamicLibrary.open('libsuperscale.so');

  static final _open = _lib.lookupFunction<_OpenNative, _Open>(
    'sc_engine_open',
  );
  static final _run = _lib.lookupFunction<_RunNative, _Run>('sc_engine_run');
  static final _close = _lib.lookupFunction<_CloseNative, _Close>(
    'sc_engine_close',
  );
  static final _free = _lib.lookupFunction<_FreeNative, _Free>(
    'sc_engine_free_string',
  );

  static String? _take(Pointer<Pointer<Uint8>> err) {
    final ptr = err.value;
    if (ptr.address == 0) return null;
    try {
      return ptr.cast<Utf8>().toDartString();
    } finally {
      _free(ptr);
    }
  }
}

class Session {
  Session._(this._handle, this.scale, this.maxSpan);

  final Pointer<Void> _handle;
  final int scale;
  final int maxSpan;
  bool _closed = false;

  static Session open({
    required String modelPath,
    required Backend backend,
    required int threads,
  }) {
    final path = modelPath.toNativeUtf8();
    final scale = calloc<Int32>();
    final maxSpan = calloc<Int32>();
    final err = calloc<Pointer<Uint8>>();
    try {
      final handle = Engine._open(
        path,
        backend.index,
        threads,
        scale,
        maxSpan,
        err,
      );
      if (handle.address == 0) {
        throw EngineFailure(Engine._take(err) ?? 'unknown error');
      }
      return Session._(handle, scale.value, maxSpan.value);
    } finally {
      calloc.free(path);
      calloc.free(scale);
      calloc.free(maxSpan);
      calloc.free(err);
    }
  }

  ({int width, int height}) run({
    required Pointer<Uint8> rgba,
    required int width,
    required int height,
    required Pointer<Uint8> out,
    required int outCapacity,
  }) {
    if (_closed) throw const EngineFailure('session closed');
    final outWidth = calloc<Int32>();
    final outHeight = calloc<Int32>();
    final err = calloc<Pointer<Uint8>>();
    try {
      final rc = Engine._run(
        _handle,
        rgba,
        width,
        height,
        out,
        outCapacity,
        outWidth,
        outHeight,
        err,
      );
      if (rc != 0) {
        throw EngineFailure(Engine._take(err) ?? 'unknown error');
      }
      return (width: outWidth.value, height: outHeight.value);
    } finally {
      calloc.free(outWidth);
      calloc.free(outHeight);
      calloc.free(err);
    }
  }

  void close() {
    if (_closed) return;
    _closed = true;
    Engine._close(_handle);
  }
}

int resolveThreads() {
  final cores = Platform.numberOfProcessors;
  return cores > 2 ? cores - 1 : cores;
}
