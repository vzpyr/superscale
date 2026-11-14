import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../models.dart';
import 'cancel.dart';

class ModelCache {
  static Directory? _directory;

  static Future<Directory> directory() async {
    final cached = _directory;
    if (cached != null) return cached;
    final support = await getApplicationSupportDirectory();
    final models = Directory('${support.path}/models');
    await models.create(recursive: true);
    _directory = models;
    return models;
  }

  static Future<File> fileFor(UpscaleModel model) async {
    final dir = await directory();
    return File('${dir.path}/${Uri.parse(model.url).pathSegments.last}');
  }

  static Future<bool> isCached(UpscaleModel model) async {
    final file = await fileFor(model);
    if (!await file.exists()) return false;
    return await file.length() == model.bytes;
  }

  static Future<File> ensure(
    UpscaleModel model, {
    required void Function(int received, int total) onProgress,
    required bool Function() isCancelled,
  }) async {
    final target = await fileFor(model);
    if (await isCached(model)) return target;

    final partial = File('${target.path}.part');
    if (await partial.exists()) await partial.delete();

    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(model.url));
      final response = await request.close();
      if (response.statusCode != HttpStatus.ok) {
        throw StateError('server returned ${response.statusCode}');
      }
      final total = response.contentLength;
      final sink = partial.openWrite();
      var received = 0;
      try {
        await for (final chunk in response) {
          if (isCancelled()) {
            throw const UpscaleCancelled();
          }
          sink.add(chunk);
          received += chunk.length;
          onProgress(received, total);
        }
      } finally {
        await sink.close();
      }
      if (await partial.length() != model.bytes) {
        await partial.delete();
        throw StateError('downloaded size mismatch');
      }
      await partial.rename(target.path);
      return target;
    } catch (_) {
      if (await partial.exists()) await partial.delete();
      rethrow;
    } finally {
      client.close();
    }
  }
}
