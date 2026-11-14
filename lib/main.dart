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
    final schemes = await SystemDynamicColorService.resolveDynamicSchemes();
    if (mounted && (schemes.light != null || schemes.dark != null)) {
      setState(() {
        _dynamicLight = schemes.light;
        _dynamicDark = schemes.dark;
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
