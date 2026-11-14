import 'dart:io';

import 'package:flutter/material.dart';

class CompareView extends StatefulWidget {
  const CompareView({
    super.key,
    required this.before,
    required this.after,
    required this.width,
    required this.height,
  });

  final File before;
  final File after;
  final int width;
  final int height;

  @override
  State<CompareView> createState() => _CompareViewState();
}

class _CompareViewState extends State<CompareView> {
  double _fraction = 0.5;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final area = Size(constraints.maxWidth, constraints.maxHeight);
        final fitted = applyBoxFit(
          BoxFit.contain,
          Size(widget.width.toDouble(), widget.height.toDouble()),
          area,
        );
        final destination = Alignment.center.inscribe(
          fitted.destination,
          Offset.zero & area,
        );

        return GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTapDown: (details) =>
              _moveTo(details.localPosition.dx, destination),
          onHorizontalDragUpdate: (details) =>
              _moveTo(details.localPosition.dx, destination),
          child: Stack(
            children: [
              Positioned.fromRect(
                rect: destination,
                child: Image.file(widget.before, fit: BoxFit.fill),
              ),
              Positioned.fromRect(
                rect: destination,
                child: ClipRect(
                  clipper: _RightClipper(_fraction),
                  child: Image.file(widget.after, fit: BoxFit.fill),
                ),
              ),
              Positioned(
                left: destination.left + destination.width * _fraction,
                top: destination.top,
                bottom: area.height - destination.bottom,
                width: 0,
                child: const OverflowBox(
                  minWidth: 0,
                  maxWidth: double.infinity,
                  alignment: Alignment.center,
                  child: VerticalDivider(),
                ),
              ),
              Positioned(
                left: destination.left + destination.width * _fraction,
                top: destination.top,
                bottom: area.height - destination.bottom,
                width: 0,
                child: OverflowBox(
                  minWidth: 0,
                  maxWidth: double.infinity,
                  minHeight: 0,
                  maxHeight: double.infinity,
                  alignment: Alignment.center,
                  child: IconButton.filled(
                    onPressed: () {},
                    icon: const Icon(Icons.drag_handle),
                    tooltip: 'Drag to compare',
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _moveTo(double dx, Rect destination) {
    final next = ((dx - destination.left) / destination.width).clamp(0.0, 1.0);
    if (next == _fraction) return;
    setState(() {
      _fraction = next;
    });
  }
}

class _RightClipper extends CustomClipper<Rect> {
  const _RightClipper(this.fraction);

  final double fraction;

  @override
  Rect getClip(Size size) {
    return Rect.fromLTWH(
      size.width * fraction,
      0,
      size.width * (1 - fraction),
      size.height,
    );
  }

  @override
  bool shouldReclip(_RightClipper oldClipper) =>
      oldClipper.fraction != fraction;
}
