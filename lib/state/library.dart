import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../engine/cache.dart';
import '../engine/cancel.dart';
import '../engine/image_io.dart';
import '../engine/native.dart';
import '../engine/worker.dart';
import '../models.dart';

enum RunMode { selected, all }

class LibraryImage {
  LibraryImage({
    required this.id,
    required this.name,
    required this.file,
    required this.bytes,
    required this.width,
    required this.height,
  });

  final String id;
  final String name;
  final File file;
  final int bytes;
  final int width;
  final int height;
  File? output;
  int? outputWidth;
  int? outputHeight;
  int? outputBytes;

  bool get upscaled => output != null;

  String get sizeLabel => upscaled
      ? '${formatBytes(bytes)} → ${formatBytes(outputBytes!)}'
      : formatBytes(bytes);
}

class LibraryStore extends ChangeNotifier {
  final List<LibraryImage> images = [];
  final Set<String> downloadedModels = {};

  String? selectedId;
  UpscaleModel model = models.firstWhere((m) => m.id == 'upscayl-standard-4x');
  RunMode? runMode;
  Backend backend = Backend.cpu;
  bool cancelling = false;
  String? status;
  double? progress;

  Directory? _imageDir;
  Directory? _outputDir;
  int _sequence = 0;
  UpscaleWorker? _worker;
  String? _pendingMessage;

  bool get busy => runMode != null;

  LibraryImage? get selected {
    final id = selectedId;
    if (id == null) return null;
    for (final image in images) {
      if (image.id == id) return image;
    }
    return null;
  }

  bool get canUpscaleSelected =>
      !busy && selected != null && !selected!.upscaled;

  bool get canUpscaleAll => !busy && images.any((image) => !image.upscaled);

  Future<void> load() async {
    final support = await getApplicationSupportDirectory();
    _imageDir = Directory('${support.path}/images');
    _outputDir = Directory('${support.path}/outputs');
    await _imageDir!.create(recursive: true);
    await _outputDir!.create(recursive: true);

    final found = <LibraryImage>[];
    final entries = await _imageDir!.list().toList();
    entries.sort((a, b) => b.path.compareTo(a.path));
    for (final entry in entries) {
      if (entry is! File) continue;
      final stored = entry.uri.pathSegments.last;
      final cut = stored.indexOf('_');
      if (cut <= 0) continue;
      final id = stored.substring(0, cut);
      final name = stored.substring(cut + 1);
      try {
        final size = await imageSize(entry);
        final output = File('${_outputDir!.path}/$id.png');
        final image = LibraryImage(
          id: id,
          name: name,
          file: entry,
          bytes: await entry.length(),
          width: size.width,
          height: size.height,
        );
        if (await output.exists()) {
          final outSize = await imageSize(output);
          image.output = output;
          image.outputWidth = outSize.width;
          image.outputHeight = outSize.height;
          image.outputBytes = await output.length();
        }
        found.add(image);
      } catch (_) {
        continue;
      }
    }

    images
      ..clear()
      ..addAll(found);
    selectedId = images.isEmpty ? null : images.first.id;

    for (final candidate in models) {
      if (await ModelCache.isCached(candidate)) {
        downloadedModels.add(candidate.id);
      }
    }

    notifyListeners();
  }

  Future<void> addImages() async {
    if (busy) return;
    final picked = await ImagePicker().pickMultiImage();
    if (picked.isEmpty) return;

    final added = <String>[];
    for (final item in picked) {
      try {
        final id = _newId();
        final name = item.name;
        final target = File('${_imageDir!.path}/${id}_$name');
        await item.saveTo(target.path);
        final size = await imageSize(target);
        images.insert(
          0,
          LibraryImage(
            id: id,
            name: name,
            file: target,
            bytes: await target.length(),
            width: size.width,
            height: size.height,
          ),
        );
        added.add(id);
      } catch (_) {
        continue;
      }
    }
    if (added.isNotEmpty) selectedId = added.first;
    notifyListeners();
  }

  void select(String id) {
    if (busy || selectedId == id) return;
    selectedId = id;
    notifyListeners();
  }

  void chooseModel(UpscaleModel next) {
    if (busy) return;
    model = next;
    notifyListeners();
  }

  Future<void> remove(String id) async {
    if (busy) return;
    final index = images.indexWhere((item) => item.id == id);
    if (index < 0) return;
    final image = images[index];
    images.removeAt(index);
    if (selectedId == id) selectedId = images.isEmpty ? null : images.first.id;
    notifyListeners();
    try {
      await image.file.delete();
      await image.output?.delete();
    } catch (_) {}
  }

  void cancel() {
    if (!busy || cancelling) return;
    cancelling = true;
    status = 'Cancelling';
    _worker?.cancel();
    notifyListeners();
  }

  Future<void> upscaleSelected() async {
    final image = selected;
    if (image == null || image.upscaled) return;
    await _upscale([image], RunMode.selected);
  }

  Future<void> upscaleAll() async {
    await _upscale(
      images.where((image) => !image.upscaled).toList(),
      RunMode.all,
    );
  }

  bool get canSaveSelected => selected?.upscaled ?? false;

  bool get canSaveAll => images.where((image) => image.upscaled).length > 1;

  Future<void> saveSelected() async {
    final image = selected;
    if (image == null) return;
    await saveToGallery(image);
  }

  Future<void> shareSelected() async {
    final image = selected;
    if (image == null) return;
    await share(image);
  }

  Future<void> saveToGallery(LibraryImage image) async {
    if (image.output == null) return;
    try {
      if (!await _ensureGalleryAccess()) {
        _pendingMessage = 'Storage permission denied';
        notifyListeners();
        return;
      }
      await _putInGallery(image);
      _pendingMessage = 'Saved to gallery';
      notifyListeners();
    } catch (error) {
      _pendingMessage = 'Could not save image: $error';
      notifyListeners();
    }
  }

  Future<void> saveAllToGallery() async {
    if (!canSaveAll) return;
    final batch = images.where((image) => image.upscaled).toList();
    try {
      if (!await _ensureGalleryAccess()) {
        _pendingMessage = 'Storage permission denied';
        notifyListeners();
        return;
      }
      var saved = 0;
      for (final image in batch) {
        await _putInGallery(image);
        saved++;
      }
      _pendingMessage = saved == 1
          ? 'Saved 1 image to gallery'
          : 'Saved $saved images to gallery';
      notifyListeners();
    } catch (error) {
      _pendingMessage = 'Could not save images: $error';
      notifyListeners();
    }
  }

  Future<bool> _ensureGalleryAccess() async {
    if (await Gal.hasPermission()) return true;
    return Gal.requestPermission();
  }

  Future<void> _putInGallery(LibraryImage image) async {
    final output = image.output;
    if (output == null) return;
    final temp = File(
      '${(await getTemporaryDirectory()).path}/${exportName(image)}',
    );
    await output.copy(temp.path);
    try {
      await Gal.putImage(temp.path, album: 'Superscale');
    } finally {
      if (await temp.exists()) await temp.delete();
    }
  }

  Future<void> share(LibraryImage image) async {
    final output = image.output;
    if (output == null) return;
    try {
      await SharePlus.instance.share(
        ShareParams(
          files: [XFile(output.path, mimeType: 'image/png')],
          fileNameOverrides: [exportName(image)],
        ),
      );
    } catch (error) {
      _pendingMessage = 'Could not share image: $error';
      notifyListeners();
    }
  }

  String exportName(LibraryImage image) {
    final dot = image.name.lastIndexOf('.');
    final base = dot < 0 ? image.name : image.name.substring(0, dot);
    return '${base}_upscaled.png';
  }

  void chooseBackend(Backend value) {
    if (busy || backend == value) return;
    backend = value;
    notifyListeners();
  }

  String? takeMessage() {
    final message = _pendingMessage;
    _pendingMessage = null;
    return message;
  }

  Future<void> _upscale(List<LibraryImage> queue, RunMode mode) async {
    if (busy || queue.isEmpty) return;
    runMode = mode;
    cancelling = false;
    notifyListeners();

    var cancelled = false;
    for (final image in queue) {
      try {
        await _upscaleOne(image);
      } on UpscaleCancelled {
        cancelled = true;
        break;
      } catch (error) {
        _pendingMessage = error is _StageError
            ? error.message
            : 'Could not upscale image: $error';
        break;
      }
    }

    if (cancelled) _pendingMessage = 'Upscale cancelled';
    runMode = null;
    cancelling = false;
    status = null;
    progress = null;
    notifyListeners();
  }

  Future<void> _upscaleOne(LibraryImage image) async {
    status = 'Loading model';
    progress = null;
    _emit();

    final File weights;
    try {
      weights = await ModelCache.ensure(
        model,
        isCancelled: () => cancelling,
        onProgress: (received, total) {
          final percent = total > 0 ? (received * 100 / total).round() : 0;
          status = 'Downloading model $percent %';
          progress = total > 0 ? received / total : null;
          _emit();
        },
      );
    } on UpscaleCancelled {
      rethrow;
    } catch (error) {
      throw _StageError('Could not download model: $error');
    }
    downloadedModels.add(model.id);

    final SourceImage source;
    try {
      source = await decodeImage(image.file);
    } catch (error) {
      throw _StageError('Could not open image: $error');
    }

    status = 'Loading model';
    progress = null;
    _emit();

    final UpscaleOutput result;
    try {
      final worker = _worker ??= await UpscaleWorker.instance();
      result = await worker.upscale(
        backend: backend,
        modelPath: weights.path,
        threads: resolveThreads(),
        pixels: source.pixels,
        width: source.width,
        height: source.height,
        onProgress: (done, total) {
          status = 'Tile $done/$total';
          progress = total > 0 ? done / total : null;
          _emit();
        },
      );
    } on UpscaleCancelled {
      rethrow;
    } catch (error) {
      throw _StageError(
        'Could not upscale image: ${error is UpscaleFailure ? error.reason : error}',
      );
    }

    status = 'Saving result';
    progress = null;
    _emit();

    final target = File('${_outputDir!.path}/${image.id}.png');
    try {
      await writePng(
        pixels: result.pixels,
        width: result.width,
        height: result.height,
        target: target,
      );
    } catch (error) {
      throw _StageError('Could not save result: $error');
    }

    image.output = target;
    image.outputWidth = result.width;
    image.outputHeight = result.height;
    image.outputBytes = await target.length();
    status = null;
    progress = null;
    _emit();
  }

  void _emit() {
    notifyListeners();
  }

  String _newId() {
    _sequence++;
    final stamp = DateTime.now().microsecondsSinceEpoch.toRadixString(36);
    return '$stamp$_sequence';
  }
}

class _StageError implements Exception {
  const _StageError(this.message);

  final String message;
}
