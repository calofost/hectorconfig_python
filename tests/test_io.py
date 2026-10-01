from pathlib import Path

from hectorconfig_py.constants import load_constants
from hectorconfig_py.io import hector_read_tables, prepare_field


DATA = Path(__file__).parents[1] / "src" / "hectorconfig_py" / "data"


def test_example_field_is_prepared():
    tables = hector_read_tables(
        DATA / "Hexas_G23_tile_263_NOT_CONFIGURED.csv",
        DATA / "Guides_G23_tile_263_NOT_CONFIGURED.csv",
    )
    field = prepare_field(tables["tile"], tables["guide"], load_constants())
    assert len(field.target_positions) == 19
    assert len(field.standard_positions) == 2
    assert len(field.guide_positions) == 6
    assert field.tile.attrs["source_path"].endswith("Hexas_G23_tile_263_NOT_CONFIGURED.csv")
