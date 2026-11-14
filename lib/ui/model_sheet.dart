import 'package:flutter/material.dart';

import '../models.dart';
import '../state/library.dart';
import '../tokens.dart';

Future<void> showModelSheet(BuildContext context, LibraryStore store) {
  return showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (context) => ListenableBuilder(
      listenable: store,
      builder: (context, _) {
        final theme = Theme.of(context);
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: Spacing.lg,
                  vertical: Spacing.sm,
                ),
                child: Text('Model', style: theme.textTheme.titleMedium),
              ),
              Flexible(
                child: ListView(
                  shrinkWrap: true,
                  children: [
                    for (final model in models)
                      ListTile(
                        title: Text(model.id),
                        subtitle: Text(_subtitle(store, model)),
                        trailing: store.model.id == model.id
                            ? const Icon(Icons.check)
                            : null,
                        onTap: () {
                          store.chooseModel(model);
                          Navigator.pop(context);
                        },
                      ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    ),
  );
}

String _subtitle(LibraryStore store, UpscaleModel model) {
  if (store.downloadedModels.contains(model.id)) {
    return '${model.sizeLabel} (Downloaded)';
  }
  return model.sizeLabel;
}
