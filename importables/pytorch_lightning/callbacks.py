import os
import shutil
from pytorch_lightning.callbacks import Callback
from pathlib import Path

from torchinfo import summary

class FileCallback(Callback):
    def __init__(self, 
                #  model_path: Path, 
                 input_shape):
        super().__init__()
        self.input_shape = input_shape
        # self.model_path = model_path

    def on_sanity_check_end(self, trainer, pl_module):
        # --- Config ---
        exp_path = Path(trainer.log_dir)
        exp_path.mkdir(exist_ok=True)
        
        # # --- Model code ---
        # shutil.copy(self.model_path, experiment_folder) 

        # --- Folders ---
        figures_path = exp_path / 'figures'
        figures_path.mkdir(parents=True, exist_ok=True)

        figure_data_path = exp_path / 'figure_data'
        figure_data_path.mkdir(parents=True, exist_ok=True)
        
        # --- Model summary ---
        summary_path = os.path.join(exp_path, "model_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            model_summary = summary(pl_module, input_size=self.input_shape,
                                    col_names=["input_size", "output_size",
                                               "num_params", "params_percent", "trainable"],
                                    verbose=0)
            f.write(str(model_summary))

        print("Model summary saved.")
