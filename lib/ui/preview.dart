import 'package:flutter/material.dart';

import '../state/library.dart';
import 'compare.dart';

class Preview extends StatelessWidget {
  const Preview({super.key, required this.store});

  final LibraryStore store;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final image = store.selected;
    if (image == null) {
      return Center(
        child: Text(
          'Your images appear here',
          style: theme.textTheme.bodyLarge?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      );
    }

    final output = image.output;
    if (output == null) {
      return Image.file(image.file, fit: BoxFit.contain);
    }

    return CompareView(
      before: image.file,
      after: output,
      width: image.width,
      height: image.height,
    );
  }
}
