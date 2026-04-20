# fix_prescan_import.py — run from project root
import re

for fpath in [
    "catfuse/sparseflow/fused_conv_lif_kernel.py",
    "catfuse/sparseflow/fused_conv_bn_lif_kernel.py",
]:
    with open(fpath, "r") as f:
        code = f.read()

    # 1. Fix import: replace _build_active_group_bitmask with _build_two_stage_metadata
    code = code.replace(
        "_build_active_group_bitmask,",
        "_build_two_stage_metadata,"
    )

    # 2. Fix call site: old API had ag_mask_buf as last positional arg,
    #    new API has (ag_mask_buf, tile_class_buf) before optional kwargs
    code = code.replace(
        """    _build_active_group_bitmask(
        x_f16, N, C_IN, H, W, H_OUT, W_OUT,
        BH, BW, GH, GW,
        kernel_size, stride, padding,
        threshold, ag_mask_buf,
    )""",
        """    _build_two_stage_metadata(
        x_f16, N, C_IN, H, W, H_OUT, W_OUT,
        BH, BW, GH, GW,
        kernel_size, stride, padding, threshold,
        ag_mask_buf, None,
    )"""
    )

    with open(fpath, "w") as f:
        f.write(code)

    print(f"Fixed: {fpath}")

print("Done. Re-run: python tests/test_01_imports.py")