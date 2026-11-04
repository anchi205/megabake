"""SM100 GEMM launch and code-generation contract."""

import os


class Sm100TableGemm:
    provider_name = "sm100_tma"
    requires_table = True
    supports_streaming = False
    supports_mixed_tiles = True
    full_operand_descriptors = True
    direct_store_with_splits = True
    generic_import = ""
    table_import = (
        "from megabake.bodies.cuda.sm100.tcgen_gemm import (\n"
        "    Sm100TmaGemm,\n"
        "    build_sm100_table_descriptors as build_table_descriptors,\n"
        ")\n"
        "from megabake.bodies.cuda.sm100.persistent_gemm import Sm100PersistentGemm\n"
        "from megabake.bodies.cuda.sm90.gemm import table_descriptors\n"
    )

    def generic_definition(self, symbol, tile, cluster):
        raise ValueError("this GEMM body requires a cooperative phase-table launch")

    def table_definition(self, spec, tile, epilogues, streaming, stream_stages):
        persistent = os.environ.get("MEGABAKE_SM100_PERSISTENT") == "1"
        width = 256 if os.environ.get("MEGABAKE_SM100_PERSISTENT_WIDTH") == "256" else 128
        symbol = (
            "Sm100TmaGemm"
            if persistent and width == 128
            else "Sm100PersistentGemm"
            if persistent
            else "Sm100TmaGemm"
        )
        stages = 2 if os.environ.get("MEGABAKE_SM100_PERSISTENT_STAGES") == "2" else 1
        if persistent and width == 128 and stages != 1:
            raise ValueError("the 128-thread persistent variant requires one accumulator stage")
        return (
            f"{spec['symbol']} = {symbol}({tile[0]}, {spec['splits']}, {tile[1]}"
            + (f", acc_stages={stages}" if persistent and width == 256 else "")
            + (", warp_specialized=True" if persistent and width == 128 else "")
            + (f", epilogues=({epilogues})" if epilogues else "")
            + ")"
        )

    def source_object(self):
        if os.environ.get("MEGABAKE_SM100_PERSISTENT") == "1":
            if os.environ.get("MEGABAKE_SM100_PERSISTENT_WIDTH") != "256":
                from .tcgen_gemm import _run_tcgen_tile

                return _run_tcgen_tile
            from .persistent_gemm import _persistent_run

            return _persistent_run
        from .tcgen_gemm import _run_tcgen_tile

        return _run_tcgen_tile


TABLE_GEMM = Sm100TableGemm()
