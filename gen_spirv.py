import pathlib
import subprocess
import tempfile

COV = 16
CICH = 4
TX = 16
TY = 16
PX = 2
PY = 2
TILEX = TX * PX
TILEY = TY * PY
PD = TILEX + 2
PN = PD * PD * CICH


def build_conv() -> str:
    L = []
    add = L.append
    add("#version 450")
    add("")
    add('#include "pc.glsl"')
    add("")
    add(f"layout(local_size_x = {TX}, local_size_y = {TY}) in;")
    add("")
    add(f"const uint CICH = {CICH}u;")
    add(f"const uint PD = {PD}u;")
    add(f"const uint PN = {PN}u;")
    add(f"const uint TILEX = {TILEX}u;")
    add(f"const uint TILEY = {TILEY}u;")
    add("")
    add("layout(std430, binding = 0) readonly buffer Weights { float w[]; };")
    add("layout(std430, binding = 1) buffer Arena { float a[]; };")
    add("")
    add("shared float sp[PN];")
    add("")
    add("void main() {")
    add("  const uint co_total = pc.n0;")
    add("  const uint ci_total = pc.n1;")
    add("  const uint hi = pc.n2;")
    add("  const uint wi = pc.n3;")
    add("  const uint ho = pc.n4;")
    add("  const uint wo = pc.n5;")
    add("  const int pad = int(pc.n6);")
    add("  const uint cop = pc.n7;")
    add("  const uint c0 = gl_WorkGroupID.z * 16u;")
    add("  const bool has_bias = pc.s0[1] != 0u;")
    add("")
    add("  const uint lx = gl_LocalInvocationID.x;")
    add("  const uint ly = gl_LocalInvocationID.y;")
    add(f"  const uint lid = ly * {TX}u + lx;")
    add("  const int base_x = int(gl_WorkGroupID.x * TILEX) - pad;")
    add("  const int base_y = int(gl_WorkGroupID.y * TILEY) - pad;")
    add(f"  const uint px0 = lx * {PX}u;")
    add(f"  const uint py0 = ly * {PY}u;")
    add("")
    for q in ("00", "01", "10", "11"):
        for j in range(COV):
            add(f"  float acc{q}_{j} = 0.0;")
    add("")
    add("  for (uint ci0 = 0u; ci0 < ci_total; ci0 += CICH) {")
    add("    const uint cn = min(CICH, ci_total - ci0);")
    add(f"    for (uint i = lid; i < PN; i += {TX * TY}u) {{")
    add("      const uint ci = i / (PD * PD);")
    add("      const uint r = i - ci * (PD * PD);")
    add("      const uint py = r / PD;")
    add("      const uint px = r - py * PD;")
    add("      float v = 0.0;")
    add("      if (ci < cn) {")
    add("        const int sx = base_x + int(px);")
    add("        const int sy = base_y + int(py);")
    add("        if (sx >= 0 && sy >= 0 && sx < int(wi) && sy < int(hi)) {")
    add("          v = a[pc.in_off + (ci0 + ci) * hi * wi + uint(sy) * wi + uint(sx)];")
    add("        }")
    add("      }")
    add("      sp[i] = v;")
    add("    }")
    add("    barrier();")
    add("    for (uint ci = 0u; ci < cn; ++ci) {")
    add("      const uint sbase = ci * (PD * PD);")
    for ky in range(3):
        for kx in range(3):
            t = ky * 3 + kx
            add(f"      const uint wb{t} = pc.w_off + ((ci0 + ci) * 9u + {t}u) * cop + c0;")
            for j in range(COV):
                add(f"      const float wv{t}_{j} = w[wb{t} + {j}u];")
            for q, pyo, pxo in (("00", 0, 0), ("01", 0, 1), ("10", 1, 0), ("11", 1, 1)):
                add(
                    f"      const float p{q}_{t} = sp[sbase + (py0 + {ky + pyo}u) * PD + px0 + {kx + pxo}u];"
                )
            for q in ("00", "01", "10", "11"):
                for j in range(COV):
                    add(f"      acc{q}_{j} = fma(wv{t}_{j}, p{q}_{t}, acc{q}_{j});")
    add("    }")
    add("    barrier();")
    add("  }")
    add("")
    add("  const uint ox = gl_WorkGroupID.x * TILEX + px0;")
    add("  const uint oy = gl_WorkGroupID.y * TILEY + py0;")
    add("  const uint obase = pc.out_off + oy * wo + ox;")
    for j in range(COV):
        add(f"  if (c0 + {j}u < co_total) {{")
        add(f"    const float bias_{j} = has_bias ? w[pc.b_off + c0 + {j}u] : 0.0;")
        add(f"    const uint dst_{j} = obase + (c0 + {j}u) * ho * wo;")
        add(f"    if (ox < wo && oy < ho) a[dst_{j}] = acc00_{j} + bias_{j};")
        add(f"    if (ox + 1u < wo && oy < ho) a[dst_{j} + 1u] = acc01_{j} + bias_{j};")
        add(f"    if (ox < wo && oy + 1u < ho) a[dst_{j} + wo] = acc10_{j} + bias_{j};")
        add(f"    if (ox + 1u < wo && oy + 1u < ho) a[dst_{j} + wo + 1u] = acc11_{j} + bias_{j};")
        add("  }")
    add("}")
    add("")
    return "\n".join(L)


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    src = root / "android" / "app" / "src" / "main" / "cpp" / "gpu"
    conv_comp = src / "shaders" / "conv.comp"
    conv_comp.write_text(build_conv())

    names = ["conv", "unary", "binary", "resize", "d2s"]
    lines = [
        "#ifndef SUPERSCALE_SPIRV_H",
        "#define SUPERSCALE_SPIRV_H",
        "",
        "#include <stdint.h>",
        "",
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        for name in names:
            spv = tmp / f"{name}.spv"
            subprocess.run(
                [
                    "glslc",
                    "--target-env=vulkan1.0",
                    "-O",
                    f"-I{src}",
                    "-o",
                    str(spv),
                    str(src / "shaders" / f"{name}.comp"),
                ],
                check=True,
            )
            subprocess.run(["spirv-val", str(spv)], check=True)

            data = spv.read_bytes()
            words = [
                int.from_bytes(data[i : i + 4], "little")
                for i in range(0, len(data), 4)
            ]
            lines.append(f"static const uint32_t sc_spv_{name}[] = {{")
            for i in range(0, len(words), 12):
                lines.append(
                    "    "
                    + ", ".join(f"0x{w:08x}u" for w in words[i : i + 12])
                    + ","
                )
            lines.append("};")
            lines.append(f"static const uint32_t sc_spv_{name}_words = {len(words)}u;")
            lines.append("")

    lines.append("#endif")
    dst = src / "spirv.h"
    dst.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
