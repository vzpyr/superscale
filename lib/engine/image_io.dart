import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

class SourceImage {
  const SourceImage({
    required this.pixels,
    required this.width,
    required this.height,
  });

  final Uint8List pixels;
  final int width;
  final int height;
}

Future<({int width, int height})> imageSize(File file) async {
  final bytes = await file.readAsBytes();
  final buffer = await ui.ImmutableBuffer.fromUint8List(bytes);
  final descriptor = await ui.ImageDescriptor.encoded(buffer);
  final result = (width: descriptor.width, height: descriptor.height);
  descriptor.dispose();
  buffer.dispose();
  return result;
}

Future<SourceImage> decodeImage(File file) async {
  final bytes = await file.readAsBytes();
  final buffer = await ui.ImmutableBuffer.fromUint8List(bytes);
  final descriptor = await ui.ImageDescriptor.encoded(buffer);
  final codec = await descriptor.instantiateCodec();
  final frame = await codec.getNextFrame();
  final width = frame.image.width;
  final height = frame.image.height;
  final data = await frame.image.toByteData(format: ui.ImageByteFormat.rawRgba);
  frame.image.dispose();
  codec.dispose();
  descriptor.dispose();
  buffer.dispose();
  if (data == null) throw StateError('image decoding failed');
  return SourceImage(
    pixels: data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
    width: width,
    height: height,
  );
}

Future<void> writePng({
  required Uint8List pixels,
  required int width,
  required int height,
  required File target,
}) async {
  final buffer = await ui.ImmutableBuffer.fromUint8List(pixels);
  final descriptor = ui.ImageDescriptor.raw(
    buffer,
    width: width,
    height: height,
    pixelFormat: ui.PixelFormat.rgba8888,
  );
  final codec = await descriptor.instantiateCodec();
  final frame = await codec.getNextFrame();
  final png = await frame.image.toByteData(format: ui.ImageByteFormat.png);
  frame.image.dispose();
  codec.dispose();
  descriptor.dispose();
  buffer.dispose();
  if (png == null) throw StateError('image encoding failed');
  await target.writeAsBytes(
    png.buffer.asUint8List(png.offsetInBytes, png.lengthInBytes),
    flush: true,
  );
}
