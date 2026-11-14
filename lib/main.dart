import 'package:flutter/material.dart';

import 'system_dynamic_color_service.dart';
import 'theme.dart';
import 'ui/home.dart';

void main() {
  runApp(const SuperscaleApp());
}

class SuperscaleApp extends StatefulWidget {
  const SuperscaleApp({super.key});

  @override
  State<SuperscaleApp> createState() => _SuperscaleAppState();
}

class _SuperscaleAppState extends State<SuperscaleApp> {
  ColorScheme? _dynamicLight;
  ColorScheme? _dynamicDark;

  @override
  void initState() {
    super.initState();
    _loadDynamicColors();
  }

  Future<void> _loadDynamicColors() async {
    final light = await SystemDynamicColorService.getDynamicColorScheme(
      brightness: Brightness.light,
    );
    final dark = await SystemDynamicColorService.getDynamicColorScheme(
      brightness: Brightness.dark,
    );
    if (mounted && (light != null || dark != null)) {
      setState(() {
        _dynamicLight = light;
        _dynamicDark = dark;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final lightScheme = SystemDynamicColorService.resolveScheme(
      dynamicScheme: _dynamicLight,
      brightness: Brightness.light,
    );
    final darkScheme = SystemDynamicColorService.resolveScheme(
      dynamicScheme: _dynamicDark,
      brightness: Brightness.dark,
    );

    return MaterialApp(
      title: 'Superscale',
      debugShowCheckedModeBanner: false,
      theme: SuperscaleTheme.createTheme(lightScheme),
      darkTheme: SuperscaleTheme.createTheme(darkScheme),
      home: const HomePage(),
    );
  }
}
