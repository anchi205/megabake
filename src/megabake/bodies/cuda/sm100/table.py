"""SM100 GEMM launch and code-generation contract."""


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
        "from megabake.bodies.cuda.sm90.gemm import table_descriptors\n"
    )

    def generic_definition(self, symbol, tile, cluster):
        raise ValueError("this GEMM body requires a cooperative phase-table launch")

    def table_definition(self, spec, tile, epilogues, streaming, stream_stages):
        return (
            f"{spec['symbol']} = Sm100TmaGemm({tile[0]}, {spec['splits']}, {tile[1]}"
            + (f", epilogues=({epilogues})" if epilogues else "")
            + ")"
        )

    def source_object(self):
        from .tcgen_gemm import _run_tcgen_tile

        return _run_tcgen_tile


TABLE_GEMM = Sm100TableGemm()
