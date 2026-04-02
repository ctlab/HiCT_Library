# HiCT Python library (JVM API-first)

This repository now provides a JVM-backed Python API as the primary and maintained interface.
Heavy operations are executed in `HiCT_JVM`; Python acts as a fast typed client layer.

## Overview

Hi-Tree actively uses Split/Merge tree structures (Treaps) to efficiently handle contig reverse and move operations without need for overwriting 2D data.

### Features
* Support for rearrangement operations (contig/scaffold reversal and translocation);
* Support for scaffolding operations (grouping multiple contigs into scaffold and ungrouping contigs from scaffold);
* Export of assembly in FASTA format;
* Export of selection context in FASTA format;
* Import of AGP assembly description;
* Saving/loading work state into the file;
* Property tests are implemented using pytest and pytest-quickcheck.

#### W.I.P.
* The minimum assembly unit right now is **contig**, which cannot be split into parts;

## Operation instructions
You can try it by using [HiCT Server](https://github.com/ctlab/HiCT_Server) to visualize and edit Hi-C contact maps in [HiCT Web UI](https://github.com/ctlab/HiCT_WebUI).
It is recommended to use virtual environments provided by `venv` module to simplify dependency management.
This library uses HiCT format for the HiC data and you can convert Cooler's `.cool` or `.mcool` files to it using [HiCT utils](https://github.com/ctlab/HiCT_Utils)

## Documentation
- JVM API client docs: [`doc/jvm_api_v1.md`](./doc/jvm_api_v1.md)
- Legacy `ContactMatrixFacet` docs (compatibility only):
  [`doc/hict.api.ContactMatrixFacet.html`](./doc/hict.api.ContactMatrixFacet.html)

## Building from source
You can run `rebuild.sh` script in source directory which will perform static type-checking of module using mypy (it may produce error messages), build library from source and reinstall it, deleting current version.

## JVM API client (v1)

Use `hict.HiCTClient` (alias of `hict_jvm_api.HiCTJVMClient`) as the default entry point.

### Key capabilities
* Open/attach/close sessions in HiCT_JVM;
* Fetch Hi-C map regions as numpy RGBA arrays (`PNG_BY_PIXELS`) for ML pipelines;
* Run scaffolding operations via API (reverse/move/split/group/ungroup/debris);
* Run converter jobs (single and batch) and monitor status;
* Link FASTA, export FASTA selections/assembly, import/export AGP;
* Convert coordinates between BP/BINS/PIXELS with hidden-contig awareness.

### Install

```bash
pip install -e .
```

### Quick start

```python
from hict import HiCTClient, Unit

client = HiCTClient("http://localhost:5000")
session = client.open_file("build/quad/combined_ind2_4DN.hict.hdf5")
resolution = session.resolutions[0]
tile = client.fetch_region_pixels(
    start_row_px=0,
    start_col_px=0,
    rows=256,
    cols=256,
    bp_resolution=resolution,
)
px = client.convert_units(1_000_000, from_unit=Unit.BP, to_unit=Unit.PIXELS, bp_resolution=resolution)
```

### Quick links
* API docs: [`doc/jvm_api_v1.md`](./doc/jvm_api_v1.md)
* Notebooks:
  * [`notebooks/jvm_api_quickstart.ipynb`](./notebooks/jvm_api_quickstart.ipynb)
  * [`notebooks/jvm_api_pytorch_dataloader.ipynb`](./notebooks/jvm_api_pytorch_dataloader.ipynb)

### Tests
* Unit tests (mocked HTTP transport):
  * `./run_jvm_api_tests.sh`
* Optional integration tests against a real running HiCT_JVM:
  * `./run_jvm_api_optional_data_tests.sh`
