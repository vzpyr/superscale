import 'package:dynamic_color/dynamic_color.dart';
import 'package:flutter/material.dart';

import 'theme.dart';

abstract final class SystemDynamicColorService {
  static Future<({ColorScheme? light, ColorScheme? dark})>
  resolveDynamicSchemes() async {
    try {
      final accent = await DynamicColorPlugin.getAccentColor();
      if (accent != null) {
        return (
          light: ColorScheme.fromSeed(
            seedColor: accent,
            brightness: Brightness.light,
          ),
          dark: ColorScheme.fromSeed(
            seedColor: accent,
            brightness: Brightness.dark,
          ),
        );
      }
    } catch (error) {
      debugPrint('Failed to extract system accent color: $error');
    }
    return (light: null, dark: null);
  }

  static ColorScheme resolveScheme({
    ColorScheme? dynamicScheme,
    required Brightness brightness,
  }) {
    if (dynamicScheme != null) {
      return dynamicScheme;
    }
    return brightness == Brightness.dark
        ? SuperscaleTheme.darkMonochrome
        : SuperscaleTheme.lightMonochrome;
  }
}
