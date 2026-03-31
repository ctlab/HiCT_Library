# HiCT JVM API v1 Python Library

`hict_jvm_api` is a Python package that uses a running `HiCT_JVM` server as the execution backend.

## Design goals

- Keep heavy matrix/assembly logic in JVM.
- Expose a Python API suitable for bioinformatics scripting and ML pipelines.
- Keep I/O efficient through pooled HTTP connections and direct region extraction (`PNG_BY_PIXELS`).

## Main classes

- `hict_jvm_api.client.HiCTJVMClient`
  - session management (`open_file`, `attach_session`, `close_session`)
  - map region fetch (`fetch_region_pixels`, `fetch_tile_png`, `fetch_tile_with_ranges`)
  - scaffolding operations (`reverse_selection_range`, `move_selection_range`, `split_contig_at_bin`, etc.)
  - conversion jobs (`start_conversion_job`, `start_batch_conversion_jobs`, polling helpers)
  - FASTA/AGP operations (`link_fasta`, `export_fasta_for_selection`, `load_agp`)
- `hict_jvm_api.units.UnitConverter`
  - fast local conversion between `BP`, `BINS`, `PIXELS` at a selected resolution,
    respecting hidden contigs via `contigPresenceAtResolution`.
- `hict_jvm_api.dataloader.HiCTRegionDataset`
  - PyTorch-friendly random-access dataset fetching regions from a live session.

## Installation

From repository root:

```bash
pip install -e .
```

With PyTorch extras:

```bash
pip install -e '.[torch]'
```

## Quick start

```python
from hict_jvm_api import HiCTJVMClient, Unit

client = HiCTJVMClient("http://localhost:5001")
open_resp = client.open_file("build/quad/combined_ind2_4DN.hict.hdf5")

resolution = open_resp.resolutions[0]  # coarse level
img = client.fetch_region_pixels(
    start_row_px=0,
    start_col_px=0,
    rows=256,
    cols=256,
    bp_resolution=resolution,
)

# Convert BP -> visible pixel coordinate
px = client.convert_units(1_000_000, from_unit=Unit.BP, to_unit=Unit.PIXELS, bp_resolution=resolution)
```

## Testing

- Unit tests (mocked transport):

```bash
./run_jvm_api_tests.sh
```

- Optional integration tests against a real server and optional files:

```bash
export HICT_JVM_API_BASE_URL=http://localhost:5001
export HICT_DATASET_FILE=build/quad/combined_ind2_4DN.hict.hdf5
# Optional:
# export HICT_FASTA_FILE=build/quad/quad_combined_ind2.fasta
# export HICT_AGP_FILE=build/quad/some.agp
# export HICT_JVM_API_ALLOW_MUTATION=true
./run_jvm_api_optional_data_tests.sh
```

## Notebook examples

See:

- `notebooks/jvm_api_quickstart.ipynb`
- `notebooks/jvm_api_pytorch_dataloader.ipynb`
