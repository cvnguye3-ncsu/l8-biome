# Landsat 8 Biome 

## Problem

Semantic image segmentation for (thick) clouds, thin clouds, and cloud shadows.

## Data

[L8-Biome](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)

1. Download raw data and copy folder structure.
2. Generate images and labels with `create_images.py`.
3. Tabulate metadata for each image with `tabulate_metadata.py`.

### Data Structure

FMask folder is optional if you do not care for it.

```
_data/
├── dataset/
│   ├── img/
│   │   ├── img_1.png
│   │   ├── img_2.png
│   │   └── . . .
│   └── label/
│       ├── img_1.png
│       ├── img_2.png
│       └── . . .
├── auxiliary/
│   └── fmask/
│       ├── img_1.png
│       ├── img_2.png
│       └── . . .
├── raw/
│   └── BC/
│       ├── LC80010112014080LGN00/
│       │   ├── LC80010112014080LGN00_B1.TIF
│       │   ├── LC80010112014080LGN00_B2.TIF
│       │   ├── LC80010112014080LGN00_B3.TIF
│       │   └── ...
│       └── LC80010732013109LGN00/
│           ├── LC80010732013109LGN00_B1.TIF
│           ├── LC80010732013109LGN00_B2.TIF
│           ├── LC80010732013109LGN00_B3.TIF
│           └── ...
└── seeds/
    ├── (seed)_(subset_ratio)_class_balance.pt
    ├── (seed)_(subset_ratio)_mean.pt
    ├── (seed)_(subset_ratio)_std.pt
    └── (seed)_(subset_ratio)_split.csv
```

- `dataset/`: Where your main imagery for training lies. 
- `BC/`: Full GeoTIFF imagery. Must follow same folder structure in the zip files. These can be compressed to half the space. Use `scripts/data_creation/reproject.py`.
- `seeds/`: Metadata for training. The class balance for loss weighing, computed mean and std for dataset, and the training/validating/testing split as a CSV (from a pandas Dataframe).

## Models

All model code taken from their respective repositories. Each neural network was rewritten (equivalent results) to be compilable with `pytorch.jit.script`; some model code needed only minor changes, such as typehints (e.g., Prithvi), while others required substanial changes - for example, turning einop operations into pure Pytorch operations (e.g., LSKNet).

- FMask 3.3
   - [Main repository](https://github.com/ubarsc/python-fmask)
   - [Documentation](https://www.pythonfmask.org/en/latest/)
- HQ-SAM
  - [Main repository](https://github.com/SysCV/sam-hq/tree/main)
- Prithvi
  - [Hugging face](https://github.com/NASA-IMPACT/hls-foundation-os/blob/main/geospatial_fm/geospatial_fm.py#L103)
  - [Segmentation example](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
- SatlasNet
  - [Main repository](https://github.com/allenai/satlas)
  - [Pretrained model repository](https://github.com/allenai/satlaspretrain_models)
- LSKNet (+UnetFormer)
  - [Main repository](https://github.com/zcablii/LSKNet)
  - [With UnetFormer](https://github.com/zcablii/GeoSeg)

### Setup

Each model lies in `models/`. Every model needs a `main.py` to run experiments. 

- FMask: Run `/scripts/fmask/main.py` for running predictions. Buffered predictions is set to 0.

Here are the weight names. Weights are easily assessible from respective repositories.

- HQ-SAM: Weights `sam_hq_vit_b.pth`.
- Prithvi: Weights `Prithvi_100M.pt`.
- SatlasNet: Weights `landsat_swinb_si.pth`.
- LSKNet: Weights `lsk_s_backbone.pth`.


## Environment

```
mamba create -n l8-biome python==3.10.* gdal==3.10.*
pip install hydra-core==1.3.* pillow opencv-python rasterio pandas matplotlib seaborn randomgen jupyter scipy tqdm 
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install lightning torchinfo timm kornia==0.8.1 segment-anything-hq
pip install git+https://github.com/ubarsc/rios
pip install git+https://github.com/ubarsc/python-fmask
```

You most likely don't need the exact versions, just matching the major versions would suffice. Becareful with `pip`; if it wants to overwrite PyTorch files, you must cancel and find another way to install your package (e.g., `conda`).

Run `pip install -e .` to install my local packages.

### Config Files
Within `conf/`, you need a `main_config.yaml` for the main hyperparameters. 

- Set `data_folder`.
- Turn on `test_run` just to check if your program will crash. 
- Turn on `perf_eval` to produce metrics and figures.
- Every other parameter is self-explanatory by name.

 `(model).yaml` config for the model specific details. Every model config file needs a '`weights_path`. Just copy the template and correct the file paths.

 - HQ-SAM: add in `num_sample_points` for testing.
 - SatlasNet: add in `fpn` to turn on or off the Feature Pyramid Network. On by default.
