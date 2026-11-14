import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

typedef _InitNative = Int32 Function(Pointer<Pointer<Uint8>> err);
typedef _Init = int Function(Pointer<Pointer<Uint8>> err);
typedef _OpenNative = Pointer<Void> Function(
  Pointer<Utf8> path,
  Int32 threads,
  Pointer<Pointer<Uint8>> err,
);
typedef _Open = Pointer<Void> Function(
  Pointer<Utf8> path,
  int threads,
  Pointer<Pointer<Uint8>> err,
);
typedef _RunNative = Int32 Function(
  Pointer<Void> session,
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
  Pointer<Void> session,
  Pointer<Uint8> rgba,
  int width,
  int height,
  Pointer<Uint8> out,
  int outLen,
  Pointer<Int32> outWidth,
  Pointer<Int32> outHeight,
  Pointer<Pointer<Uint8>> err,
);
typedef _ScaleNative = Int32 Function(Pointer<Void> session);
typedef _Scale = int Function(Pointer<Void> session);
typedef _CloseNative = Void Function(Pointer<Void> session);
typedef _Close = void Function(Pointer<Void> session);
typedef _FreeNative = Void Function(Pointer<Uint8> text);
typedef _Free = void Function(Pointer<Uint8> text);

class OrtEngine {
  static final DynamicLibrary _lib = DynamicLibrary.open(
    'libsuperscale_ort.so',
  );

  static final _init = _lib.lookupFunction<_InitNative, _Init>(
    'superscale_init',
  );
  static final _open = _lib.lookupFunction<_OpenNative, _Open>(
    'superscale_open',
  );
  static final _run = _lib.lookupFunction<_RunNative, _Run>('superscale_run');
  static final _scale = _lib.lookupFunction<_ScaleNative, _Scale>(
    'superscale_scale',
  );
  static final _close = _lib.lookupFunction<_CloseNative, _Close>(
    'superscale_close',
  );
  static final _free = _lib.lookupFunction<_FreeNative, _Free>(
    'superscale_free_string',
  );

  static bool _started = false;

  static void _start() {
    if (_started) return;
    final err = malloc<Pointer<Uint8>>();
    if (_init(err) != 0) {
      final message = _takeError(err);
      malloc.free(err);
      throw StateError(message);
    }
    malloc.free(err);
    _started = true;
  }

  static String _takeError(Pointer<Pointer<Uint8>> slot) {
    final pointer = slot.value;
    if (pointer == nullptr) return 'unknown error';
    final message = pointer.cast<Utf8>().toDartString();
    _free(pointer);
    return message;
  }

  static OrtSession open({required String modelPath, required int threads}) {
    _start();
    final err = malloc<Pointer<Uint8>>();
    final path = modelPath.toNativeUtf8();
    final handle = _open(path, threads, err);
    malloc.free(path);
    if (handle == nullptr) {
      final message = _takeError(err);
      malloc.free(err);
      throw StateError(message);
    }
    malloc.free(err);
    return OrtSession._(handle);
  }
}

class OrtSession {
  OrtSession._(this._handle);

  final Pointer<Void> _handle;
  bool _closed = false;

  int get scale => OrtEngine._scale(_handle);

  ({int width, int height}) run({
    required Pointer<Uint8> rgba,
    required int width,
    required int height,
    required Pointer<Uint8> out,
    required int outCapacity,
  }) {
    final err = malloc<Pointer<Uint8>>();
    final outWidth = malloc<Int32>();
    final outHeight = malloc<Int32>();
    final code = OrtEngine._run(
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
    if (code != 0) {
      final message = OrtEngine._takeError(err);
      malloc.free(err);
      malloc.free(outWidth);
      malloc.free(outHeight);
      throw StateError(message);
    }
    final result = (width: outWidth.value, height: outHeight.value);
    malloc.free(err);
    malloc.free(outWidth);
    malloc.free(outHeight);
    return result;
  }

  void close() {
    if (_closed) return;
    _closed = true;
    OrtEngine._close(_handle);
  }
}

int resolveThreads() {
  final cores = Platform.numberOfProcessors;
  if (cores <= 1) return 1;
  return cores < 4 ? cores : 4;
}
