# Entrypoint for training when2plan

import warnings

warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated as an API", category=UserWarning
)
import hydra
from omegaconf import DictConfig, OmegaConf
from when2plan.train_loop import train_loop
from when2plan.train_loop_ablation import train_loop as train_loop_ablation
import os


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if cfg.ablation != "none":
        print(f"Running with ablation: {cfg.ablation}")
        train_loop_ablation(cfg, output_dir)
    else:
        print("Running with no ablation")
        train_loop(cfg, output_dir)

if __name__ == "__main__":
    main()
