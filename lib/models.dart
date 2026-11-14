class UpscaleModel {
  const UpscaleModel({
    required this.id,
    required this.url,
    required this.bytes,
  });

  final String id;
  final String url;
  final int bytes;

  String get sizeLabel => formatBytes(bytes);
}

String formatBytes(int bytes) {
  if (bytes < 1024) return '$bytes B';
  if (bytes < 1024 * 1024) return '${(bytes / 1024).round()} kB';
  if (bytes < 1024 * 1024 * 1024) {
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
  return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
}

const List<UpscaleModel> models = [
  UpscaleModel(
    id: 'upscayl-digital-art-4x',
    url: 'https://huggingface.co/huggingworld/onnx-image-models/resolve/main/upscayl-digital-art-4x.onnx',
    bytes: 17970916,
  ),
  UpscaleModel(
    id: 'upscayl-high-fidelity-4x',
    url: 'https://huggingface.co/huggingworld/onnx-image-models/resolve/main/upscayl-high-fidelity-4x.onnx',
    bytes: 67167470,
  ),
  UpscaleModel(
    id: 'upscayl-lite-4x',
    url: 'https://huggingface.co/huggingworld/onnx-image-models/resolve/main/upscayl-lite-4x.onnx',
    bytes: 4868758,
  ),
  UpscaleModel(
    id: 'upscayl-remacri-4x',
    url: 'https://huggingface.co/huggingworld/onnx-image-models/resolve/main/upscayl-remacri-4x.onnx',
    bytes: 67167470,
  ),
  UpscaleModel(
    id: 'upscayl-standard-4x',
    url: 'https://huggingface.co/huggingworld/onnx-image-models/resolve/main/upscayl-standard-4x.onnx',
    bytes: 67167470,
  ),
  UpscaleModel(
    id: 'upscayl-ultramix-balanced-4x',
    url: 'https://huggingface.co/huggingworld/onnx-image-models/resolve/main/upscayl-ultramix-balanced-4x.onnx',
    bytes: 67167470,
  ),
  UpscaleModel(
    id: 'upscayl-ultrasharp-4x',
    url: 'https://huggingface.co/huggingworld/onnx-image-models/resolve/main/upscayl-ultrasharp-4x.onnx',
    bytes: 67167470,
  ),
];
