import 'package:dynamic_color/dynamic_color.dart';
import 'package:flutter/material.dart';
import 'package:material_color_utilities/material_color_utilities.dart';

import 'theme.dart';

abstract final class SystemDynamicColorService {
  static Future<ColorScheme?> getDynamicColorScheme({
    required Brightness brightness,
  }) async {
    try {
      final corePalette = await DynamicColorPlugin.getCorePalette();
      if (corePalette != null) {
        return createSchemeFromCorePalette(corePalette, brightness);
      }
    } catch (error) {
      debugPrint('Failed to extract core palette: $error');
    }

    try {
      final accentColor = await DynamicColorPlugin.getAccentColor();
      if (accentColor != null) {
        return ColorScheme.fromSeed(
          seedColor: accentColor,
          brightness: brightness,
        );
      }
    } catch (error) {
      debugPrint('Failed to extract system accent color: $error');
    }

    return null;
  }

  static ColorScheme createSchemeFromCorePalette(
    dynamic palettes,
    Brightness brightness,
  ) {
    final primaryHct = Hct.fromInt(palettes.primary.get(40) as int);
    final scheme = DynamicScheme(
      sourceColorHct: primaryHct,
      variant: Variant.tonalSpot,
      isDark: brightness == Brightness.dark,
      contrastLevel: 0.0,
      primaryPalette: palettes.primary,
      secondaryPalette: palettes.secondary,
      tertiaryPalette: palettes.tertiary,
      neutralPalette: palettes.neutral,
      neutralVariantPalette: palettes.neutralVariant,
      errorPalette: palettes.error,
    );

    return ColorScheme(
      brightness: brightness,
      primary: Color(scheme.primary),
      onPrimary: Color(scheme.onPrimary),
      primaryContainer: Color(scheme.primaryContainer),
      onPrimaryContainer: Color(scheme.onPrimaryContainer),
      secondary: Color(scheme.secondary),
      onSecondary: Color(scheme.onSecondary),
      secondaryContainer: Color(scheme.secondaryContainer),
      onSecondaryContainer: Color(scheme.onSecondaryContainer),
      tertiary: Color(scheme.tertiary),
      onTertiary: Color(scheme.onTertiary),
      tertiaryContainer: Color(scheme.tertiaryContainer),
      onTertiaryContainer: Color(scheme.onTertiaryContainer),
      error: Color(scheme.error),
      onError: Color(scheme.onError),
      errorContainer: Color(scheme.errorContainer),
      onErrorContainer: Color(scheme.onErrorContainer),
      surface: Color(scheme.surface),
      onSurface: Color(scheme.onSurface),
      surfaceContainerLowest: Color(scheme.surfaceContainerLowest),
      surfaceContainerLow: Color(scheme.surfaceContainerLow),
      surfaceContainer: Color(scheme.surfaceContainer),
      surfaceContainerHigh: Color(scheme.surfaceContainerHigh),
      surfaceContainerHighest: Color(scheme.surfaceContainerHighest),
      onSurfaceVariant: Color(scheme.onSurfaceVariant),
      outline: Color(scheme.outline),
      outlineVariant: Color(scheme.outlineVariant),
      shadow: Color(scheme.shadow),
      scrim: Color(scheme.scrim),
      inverseSurface: Color(scheme.inverseSurface),
      onInverseSurface: Color(scheme.inverseOnSurface),
      inversePrimary: Color(scheme.inversePrimary),
      surfaceDim: Color(scheme.surfaceDim),
      surfaceBright: Color(scheme.surfaceBright),
      surfaceTint: Color(scheme.primary),
      primaryFixed: Color(scheme.primaryFixed),
      primaryFixedDim: Color(scheme.primaryFixedDim),
      onPrimaryFixed: Color(scheme.onPrimaryFixed),
      onPrimaryFixedVariant: Color(scheme.onPrimaryFixedVariant),
      secondaryFixed: Color(scheme.secondaryFixed),
      secondaryFixedDim: Color(scheme.secondaryFixedDim),
      onSecondaryFixed: Color(scheme.onSecondaryFixed),
      onSecondaryFixedVariant: Color(scheme.onSecondaryFixedVariant),
      tertiaryFixed: Color(scheme.tertiaryFixed),
      tertiaryFixedDim: Color(scheme.tertiaryFixedDim),
      onTertiaryFixed: Color(scheme.onTertiaryFixed),
      onTertiaryFixedVariant: Color(scheme.onTertiaryFixedVariant),
    );
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
