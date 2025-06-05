import os
import shutil
from pytorch_lightning.callbacks import Callback
from pathlib import Path

from torchinfo import summary


class MyCallback(Callback):
    def __init__(self, model_path: Path, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.model_path = model_path

    def on_sanity_check_end(self, trainer, pl_module):
        # CONFIG
        # ------
        experiment_folder = Path(trainer.log_dir)
        experiment_folder.mkdir(exist_ok=True)
        
        # Copy model config file
        model_config_path = Path(os.getcwd())
        model_config_path = model_config_path.parent.parent / "conf" / f"{model_config_path.name}.yaml"

        fn = model_config_path.name.replace(".", "_copy.")
        outpath = experiment_folder / fn
        
        shutil.copy(model_config_path, outpath) 
        
        # Copy main config file
        main_config_path = model_config_path.parent / "main_config.yaml"
        fn = main_config_path.name.replace('.', '_copy.')
        outpath = experiment_folder / fn
        shutil.copy(main_config_path, outpath)

        # Copy model code
        shutil.copy(self.model_path, experiment_folder) 

        print("Config copied.")

        # PLOT FOLDERS
        # ------------
        plot_folder = experiment_folder / 'plot'
        plot_folder.mkdir(parents=True, exist_ok=True)

        plot_data_folder = experiment_folder / 'plot_data'
        plot_data_folder.mkdir(parents=True, exist_ok=True)
        
        # MODEL SUMMARY
        # -------------
        summary_path = os.path.join(experiment_folder, "model_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            model_summary = summary(pl_module, input_size=self.input_shape,
                                    col_names=["input_size",
                                               "output_size",
                                               "num_params",
                                               "params_percent",
                                               "trainable"],
                                    verbose=0)
            f.write(str(model_summary))

        print("Model summary saved.")
