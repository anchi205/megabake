"""Hopper GEMM launch and code-generation contract."""

from .gemm import HopperWgmmaGemmPersistentKernel


class HopperTableGemm:
    provider_name = "g0_range"
    requires_table = False
    supports_streaming = True
    supports_mixed_tiles = False
    full_operand_descriptors = False
    direct_store_with_splits = False
    generic_import = "from megabake.bodies.cuda.sm90.gemm import HopperWgmmaGemmPersistentKernel"
    table_import = (
        "from megabake.bodies.cuda.sm90.gemm import (\n"
        "    HopperWgmmaGemmPersistentKernel,\n"
        "    build_table_descriptors,\n"
        "    table_descriptors,\n"
        "    union_shared_storage,\n"
        ")\n"
    )

    def generic_definition(self, symbol, tile, cluster):
        return (
            f"{symbol} = HopperWgmmaGemmPersistentKernel(\n"
            "    cutlass.Float32,\n"
            f"    {tile!r},\n"
            f"    {cluster!r},\n"
            "    1,\n"
            "    True,\n"
            ")"
        )

    def table_definition(self, spec, tile, epilogues, streaming, stream_stages):
        return (
            f"{spec['symbol']} = HopperWgmmaGemmPersistentKernel(\n"
            "    cutlass.Float32,\n"
            f"    {tile!r},\n"
            "    (1, 1),\n"
            "    1,\n"
            "    True,\n"
            f"    smem_capacity={spec['scratch']},\n"
            "    reallocate_registers=False,\n"
            f"    splits={spec['splits']},\n"
            f"    bias_features={spec['bias_features']},\n"
            f"    bias_along_m={spec['swap_ab']!r},\n"
            f"    direct_store={spec['swap_ab'] and spec['splits'] == 1!r},\n"
            + (f"    epilogues=({epilogues}),\n" if epilogues else "")
            + ("    paired=True,\n" if spec["paired"] else "")
            + (f"    stream_stages={stream_stages},\n" if streaming else "")
            + ")"
        )

    def source_object(self):
        return HopperWgmmaGemmPersistentKernel.run


TABLE_GEMM = HopperTableGemm()
