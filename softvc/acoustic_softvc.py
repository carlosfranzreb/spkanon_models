"""
Acoustic model for the SoftVC model. It is preceded by a HuBERT model and followed
by a HiFiGAN model.
"""

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from spkanon_models.softvc.acoustic.model import AcousticModel


SAMPLE_RATE = 16000  # model's sample rate


class AcousticSoftVC:
    def __init__(self, config, device):
        """
        - The config must indicate under which key are placed the transcripts in the
            batch, under `config.input`..
        - The model is loaded following the instructions in the notebook provided in
            the repository https://github.com/bshall/soft-vc.
        """
        self.config = config
        self.device = device

        self.model = AcousticModel(False, True)
        acoustic_ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-soft-0321fd7e.pt",
            map_location="cpu",
        )
        consume_prefix_in_state_dict_if_present(
            acoustic_ckpt["acoustic-model"], "module."
        )
        self.model.load_state_dict(acoustic_ckpt["acoustic-model"])
        self.model.to(device)
        self.model.eval()

    def run(self, batch):
        """
        Given the HuBERT units, placed in the batch under the key `self.input`,
        computes and returns the spectrogram.
        """
        feats = batch[self.config.input.feats]
        n_feats = batch[self.config.input.n_feats]
        n_samples = n_feats * self.config.upsampling_ratio
        spec = self.model.generate(feats).transpose(1, 2)
        target = torch.zeros(n_samples.shape[0], dtype=torch.int32)
        return {"spectrogram": spec, "n_samples": n_samples, "target": target}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
