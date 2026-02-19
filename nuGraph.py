import hydra
import logging
from omegaconf import DictConfig, OmegaConf
# from source.train import run_training
from source.preprocess import preprocess_root_to_parquet
# from source.test import run_testing


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:

    # Set log level
    level = logging.DEBUG if cfg.logging.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    if cfg.mode.lower() == "preprocess":
        preprocess_data(cfg)

    # elif cfg.mode.lower() == "train":
    #     run_training(cfg)    
    
    # elif cfg.mode.lower()== "test":
    #     run_testing(cfg)

    else:
        logging.error(f"Unknown mode: {cfg.mode}")
    


if __name__ == "__main__":
    my_app()