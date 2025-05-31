import subprocess as sp
from pathlib import Path

RAW_PATH = Path('./_data/raw/BC/')

if __name__ == '__main__':
    for item in RAW_PATH.glob('*'):
        if not item.is_dir():
            continue
        
        scene_name = item.name
        path = str(item.resolve())
        
        print(path)
        
        print("Scene name: ", scene_name)
        
        cloud_mask = item / f'{scene_name}_fmask.img'
        
        # if cloud_mask.exists():
        #     continue
        
        sp.run(['fmask_usgsLandsatStacked', 
                '-o', str(cloud_mask.resolve()),
                '--scenedir', path,
                '--cloudbufferdistance', '0',
                '--shadowbufferdistance', '0',
                ], check=True)
        
                
                