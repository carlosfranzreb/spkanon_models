"""
Acoustic model for the SoftVC model. It is preceded by a HuBERT model and followed
by a HiFiGAN model.
"""

import logging

from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark


LOGGER = logging.getLogger("progress")
SAMPLE_RATE = 24000  # model's sample rate


class SingletonBarkVC:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(SingletonBarkVC, cls).__new__(cls)
            # Initialize your large object here, for example:
            cls._instance.bark = BarkModel().model
        return cls._instance


class BarkModel:
    def __init__(self):
        bark_cfg = BarkConfig()
        bark_cfg["REMOTE_MODEL_PATHS"]["text"][
            "path"
        ] = "https://huggingface.co/erogol/bark/resolve/main/text_2.pt?download=true"
        bark_cfg["REMOTE_MODEL_PATHS"]["coarse"][
            "path"
        ] = "https://huggingface.co/erogol/bark/resolve/main/coarse_2.pt?download=true"
        bark_cfg["REMOTE_MODEL_PATHS"]["fine"][
            "path"
        ] = "https://huggingface.co/erogol/bark/resolve/main/fine_2.pt?download=true"
        self.model = Bark.init_from_config(bark_cfg)
        self.model.load_checkpoint(
            bark_cfg, checkpoint_dir="checkpoints/bark", eval=True
        )
