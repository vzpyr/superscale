import 'package:flutter/material.dart';

import '../state/library.dart';
import '../tokens.dart';

class LibraryBar extends StatelessWidget {
  const LibraryBar({super.key, required this.store});

  final LibraryStore store;

  @override
  Widget build(BuildContext context) {
    if (store.images.isEmpty) return const SizedBox.shrink();
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: Spacing.lg),
      child: Row(
        spacing: Spacing.sm,
        children: [
          for (final image in store.images)
            GestureDetector(
              onLongPress: store.busy
                  ? null
                  : () => _confirmRemove(context, image),
              child: ChoiceChip(
                avatar: CircleAvatar(
                  backgroundImage: ResizeImage(
                    FileImage(image.file),
                    width: 96,
                  ),
                ),
                label: Text(image.sizeLabel),
                selected: store.selectedId == image.id,
                onSelected: (_) => store.select(image.id),
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _confirmRemove(BuildContext context, LibraryImage image) async {
    final remove = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Remove image?'),
        content: const Text(
          'This deletes the original and the upscaled result. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );
    if (remove == true) await store.remove(image.id);
  }
}
