# Modified from Chuning

import os
import sys
import wandb
from maze.utils.diffusion_logger import configure_logger

sys.path.append(".")
os.environ["WANDB_START_METHOD"] = "thread"


def setup_logger(config):
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "frozen-noise"),
        entity=os.environ.get("MY_WANDB_ID", None),
        group=config.env_id,
        job_type=config.algo,
        config=config,
    )
    logdir = os.path.join(
        "logdir",
        config.algo,
        config.env_id,
        config.expr_name,
        str(config.seed),
    )
    logger = configure_logger(logdir, ["stdout", "tensorboard", "wandb"])
    return logger
