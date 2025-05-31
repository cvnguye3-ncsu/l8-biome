from itertools import product
from pathlib import Path

import rasterio as rio 
from rasterio.enums import Resampling
import rasterio.warp as rio_warp
from rasterio.shutil import copy as riocopy

# Path
# ----
BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / '_data'
SCENE_PATH = DATA_PATH / 'raw' / 'BC'

BANDS = range(1, 12)
OVERWRITE = False

# Constants
# ---------
NODATA = 0

COMPRESS_OPTS = {
    "compress": "DEFLATE",  # Explorer-friendly & lossless
    "zlevel": 9,           # 1–12, 9 ≈ good ratio ⇄ speed
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "predictor": 2,          # improves integer compression
    "BIGTIFF": "IF_SAFER",  # auto‑promote if >4 GB
}

ADD_OVERVIEWS     = False           # set False to skip
OVERVIEW_FACTORS  = [2, 4, 8, 16]  # decimations; 16 ⇒ 1/16 res ≈ thumbnail
OVERVIEW_METHOD   = Resampling.average  # nearest for categorical masks

# Helper
# ------
def _extras(path: Path) -> None:
    with rio.open(path, mode="r+") as dst:
        if dst.nodata == None: dst.nodata = 0
        
        if not ADD_OVERVIEWS: return
        dst.build_overviews(OVERVIEW_FACTORS, OVERVIEW_METHOD)
        dst.update_tags(
            ns="rio_overview",
            resampling=OVERVIEW_METHOD.value
        )

# Main
# ----
if __name__ == '__main__':
    SCENES = [p for p in SCENE_PATH.iterdir() if p.is_dir()]
    
    for scene, band in product(SCENES, BANDS):
        print(scene)
        print(band)
        
        src_path = scene / f"{scene.name}_B{band}.TIF"
        dst_path = scene / f"{scene.name}_B{band}_compressed.TIF"

        with rio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update(COMPRESS_OPTS)

            riocopy(src, dst_path, **profile)
            _extras(dst_path)
            
        break