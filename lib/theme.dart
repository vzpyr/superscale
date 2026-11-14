import 'package:flutter/material.dart' as f;

abstract final class SuperscaleTheme {
  static const String fontFamily = 'Google Sans';

  static const f.ColorScheme lightMonochrome = f.ColorScheme(
    brightness: f.Brightness.light,
    primary: const f.Color(0xFF1F1F1F),
    onPrimary: const f.Color(0xFFFFFFFF),
    primaryContainer: const f.Color(0xFFE3E3E3),
    onPrimaryContainer: const f.Color(0xFF1F1F1F),
    secondary: const f.Color(0xFF5E5E5E),
    onSecondary: const f.Color(0xFFFFFFFF),
    secondaryContainer: const f.Color(0xFFEBEBEB),
    onSecondaryContainer: const f.Color(0xFF1F1F1F),
    tertiary: const f.Color(0xFF424242),
    onTertiary: const f.Color(0xFFFFFFFF),
    tertiaryContainer: const f.Color(0xFFDFDFDF),
    onTertiaryContainer: const f.Color(0xFF141414),
    error: const f.Color(0xFFBA1A1A),
    onError: const f.Color(0xFFFFFFFF),
    errorContainer: const f.Color(0xFFFFDAD6),
    onErrorContainer: const f.Color(0xFF410002),
    surface: const f.Color(0xFFF9F9F9),
    onSurface: const f.Color(0xFF1C1C1C),
    surfaceContainerLowest: const f.Color(0xFFFFFFFF),
    surfaceContainerLow: const f.Color(0xFFF3F3F3),
    surfaceContainer: const f.Color(0xFFEEEEEE),
    surfaceContainerHigh: const f.Color(0xFFE8E8E8),
    surfaceContainerHighest: const f.Color(0xFFE2E2E2),
    onSurfaceVariant: const f.Color(0xFF444746),
    outline: const f.Color(0xFF747775),
    outlineVariant: const f.Color(0xFFC4C7C5),
    shadow: const f.Color(0xFF000000),
    scrim: const f.Color(0xFF000000),
    inverseSurface: const f.Color(0xFF313030),
    onInverseSurface: const f.Color(0xFFF2F0EF),
    inversePrimary: const f.Color(0xFFC7C7C7),
  );

  static const f.ColorScheme darkMonochrome = f.ColorScheme(
    brightness: f.Brightness.dark,
    primary: const f.Color(0xFFE3E3E3),
    onPrimary: const f.Color(0xFF1F1F1F),
    primaryContainer: const f.Color(0xFF3B3B3B),
    onPrimaryContainer: const f.Color(0xFFE3E3E3),
    secondary: const f.Color(0xFFC7C7C7),
    onSecondary: const f.Color(0xFF1F1F1F),
    secondaryContainer: const f.Color(0xFF474747),
    onSecondaryContainer: const f.Color(0xFFE3E3E3),
    tertiary: const f.Color(0xFFDFDFDF),
    onTertiary: const f.Color(0xFF1F1F1F),
    tertiaryContainer: const f.Color(0xFF505050),
    onTertiaryContainer: const f.Color(0xFFDFDFDF),
    error: const f.Color(0xFFFFB4AB),
    onError: const f.Color(0xFF690005),
    errorContainer: const f.Color(0xFF93000A),
    onErrorContainer: const f.Color(0xFFFFDAD6),
    surface: const f.Color(0xFF121212),
    onSurface: const f.Color(0xFFE3E3E3),
    surfaceContainerLowest: const f.Color(0xFF0F0F0F),
    surfaceContainerLow: const f.Color(0xFF1D1D1D),
    surfaceContainer: const f.Color(0xFF232323),
    surfaceContainerHigh: const f.Color(0xFF2B2B2B),
    surfaceContainerHighest: const f.Color(0xFF363636),
    onSurfaceVariant: const f.Color(0xFFC4C7C5),
    outline: const f.Color(0xFF8E918F),
    outlineVariant: const f.Color(0xFF444746),
    shadow: const f.Color(0xFF000000),
    scrim: const f.Color(0xFF000000),
    inverseSurface: const f.Color(0xFFE3E3E3),
    onInverseSurface: const f.Color(0xFF1F1F1F),
    inversePrimary: const f.Color(0xFF5E5E5E),
  );

  static f.ThemeData createTheme(f.ColorScheme colorScheme) {
    return f.ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      fontFamily: fontFamily,
    );
  }
}
