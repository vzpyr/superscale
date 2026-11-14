import 'package:flutter/material.dart';

import '../state/library.dart';
import '../tokens.dart';
import 'library_bar.dart';
import 'model_sheet.dart';
import 'preview.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final LibraryStore _store = LibraryStore();

  @override
  void initState() {
    super.initState();
    _store.addListener(_announceMessage);
    _store.load();
  }

  @override
  void dispose() {
    _store.removeListener(_announceMessage);
    _store.dispose();
    super.dispose();
  }

  void _announceMessage() {
    final message = _store.takeMessage();
    if (message == null || !mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _store,
      builder: (context, _) => Scaffold(
        appBar: AppBar(
          title: const Text('Superscale'),
          actions: [
            if (_store.canSaveSelected) ...[
              IconButton(
                onPressed: _store.saveSelected,
                icon: const Icon(Icons.save_alt),
                tooltip: 'Save to gallery',
              ),
              IconButton(
                onPressed: _store.shareSelected,
                icon: const Icon(Icons.share_outlined),
                tooltip: 'Share',
              ),
            ],
            if (_store.canSaveAll)
              IconButton(
                onPressed: _store.saveAllToGallery,
                icon: const Icon(Icons.download_outlined),
                tooltip: 'Save all',
              ),
            IconButton(
              onPressed: _store.busy ? null : _store.addImages,
              icon: const Icon(Icons.add_photo_alternate_outlined),
              tooltip: 'Add image',
            ),
          ],
        ),
        body: SafeArea(
          child: Column(
            children: [
              Expanded(child: Preview(store: _store)),
              LibraryBar(store: _store),
              Padding(
                padding: const EdgeInsets.all(Spacing.lg),
                child: _Controls(store: _store),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Controls extends StatelessWidget {
  const _Controls({required this.store});

  final LibraryStore store;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: AlignmentDirectional.centerStart,
          child: OutlinedButton.icon(
            onPressed: store.busy ? null : () => showModelSheet(context, store),
            icon: const Icon(Icons.auto_awesome),
            label: Text(store.model.id),
          ),
        ),
        if (store.busy) ...[
          const SizedBox(height: Spacing.md),
          LinearProgressIndicator(value: store.progress),
          const SizedBox(height: Spacing.sm),
          Text(
            store.status ?? '',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
        const SizedBox(height: Spacing.md),
        Row(
          children: [
            Expanded(
              child: FilledButton(
                onPressed: store.runMode == RunMode.selected
                    ? (store.cancelling ? null : store.cancel)
                    : (store.canUpscaleSelected ? store.upscaleSelected : null),
                child: Text(
                  store.runMode == RunMode.selected ? 'Cancel' : 'Upscale',
                ),
              ),
            ),
            const SizedBox(width: Spacing.md),
            Expanded(
              child: OutlinedButton(
                onPressed: store.runMode == RunMode.all
                    ? (store.cancelling ? null : store.cancel)
                    : (store.canUpscaleAll ? store.upscaleAll : null),
                child: Text(
                  store.runMode == RunMode.all ? 'Cancel' : 'Upscale all',
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
