from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from hydra.utils import instantiate

# Here is where I import my modules
from src.data.loaders import LOFARDataModule
from src.experiment import ExperimentBase

def run_setup(config: DictConfig):
    """
    This runs some configuration setup before we start.
    :param config: Hydra config
    """
    # Set device variable
    if config.common.device == "cuda" and not torch.cuda.is_available():
        OmegaConf.update(config, "common.device", "cpu", overwrite=True)

    # Propagate device to batch collation
    from src.data.utils import update_collate
    update_collate(config.preproc.device)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Setup
    run_setup(cfg)
    # Get wandb logger from config
    wandb_logger = WandbLogger(project=cfg.project_name,log_model=True)

    # Get checkpoint callback from config
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)

    model = instantiate(cfg.model)
    experiment = ExperimentBase(model,cfg)
    data = LOFARDataModule(cfg.data,cfg.preproc,cfg.loader)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **cfg.trainer
    )

    trainer.fit(experiment, data)

if __name__ == "__main__":
    main()
