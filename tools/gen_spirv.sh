#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/android/app/src/main/cpp/gpu"
dst="$src/spirv.h"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

python3 "$root/tools/gen_conv.py" "$src/shaders/conv.comp"

for name in conv unary binary resize d2s; do
  glslc --target-env=vulkan1.0 -O -I "$src" -o "$tmp/$name.spv" "$src/shaders/$name.comp"
  spirv-val "$tmp/$name.spv"
done

python3 - "$tmp" "$dst" <<'PY'
import pathlib
import sys

tmp = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
names = ["conv", "unary", "binary", "resize", "d2s"]

lines = ["#ifndef SUPERSCALE_SPIRV_H", "#define SUPERSCALE_SPIRV_H", "", "#include <stdint.h>", ""]
for name in names:
    data = (tmp / f"{name}.spv").read_bytes()
    words = [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
    lines.append(f"static const uint32_t sc_spv_{name}[] = {{")
    for i in range(0, len(words), 12):
        lines.append("    " + ", ".join(f"0x{w:08x}u" for w in words[i : i + 12]) + ",")
    lines.append("};")
    lines.append(f"static const uint32_t sc_spv_{name}_words = {len(words)}u;")
    lines.append("")
lines.append("#endif")
dst.write_text("\n".join(lines) + "\n")
print(f"wrote {dst}")
PY
