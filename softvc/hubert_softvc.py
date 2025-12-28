"""
HuBERT component of the SoftVC model.
"""

import torch
from omegaconf import DictConfig

from spkanon_eval.datamodules import AudioBatch

SAMPLE_RATE = 16000  # model's sample rate


class HubertSoftVC:
    def __init__(self, config: DictConfig, device: str):
        """
        The model is loaded following the instructions in the notebook provided in
        the repository https://github.com/bshall/soft-vc.
        """
        self.config = config
        self.device = device

        if not hasattr(torch.nn.utils.parametrizations, "weight_norm"):
            torch.nn.utils.parametrizations.weight_norm = torch.nn.utils.weight_norm

        self.model = torch.hub.load(
            "bshall/hubert:main", "hubert_soft", force_reload=True
        )
        self.model.to(self.device)
        self.model.eval()

    def run(self, batch: AudioBatch) -> dict:
        """
        Returns the acoustic units for the given NeMo batch, which is a tuple where
        the audio batch is placed in the first position.
        """
        audio = batch.audios.unsqueeze(1)
        n_feats = batch.lens // self.config.downsampling_ratio
        feats = self.model.units(audio)
        return {"feats": feats, "n_feats": n_feats}

    def to(self, device: str):
        """Implementation of PyTorch's `to()` method to set the device."""
        self.device = device
        self.model.to(self.device)
