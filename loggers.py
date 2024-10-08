import os
import wandb
import pytz
from datetime import datetime
import pandas as pd


class WandBLogger:
    """WandB logger."""

    def __init__(self):
        self.logger = wandb.init(
            project = "llm-jail"
        )

    def log(self, dict):
        self.logger.log(dict)

    def finish(self):
        self.logger.finish()

