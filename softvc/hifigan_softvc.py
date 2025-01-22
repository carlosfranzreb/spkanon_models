import sys

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

sys.path.append("softvc_hifigan")

from spkanon_models.softvc.hifigan.generator import HifiganGenerator


class HifiganSoftVC:
    def __init__(self, config, device):
        """
        The model is loaded following the instructions in the notebook provided in
        the repository https://github.com/bshall/soft-vc.
        - The config must indicate under which key are placed the spectrograms in the
            batch, under `config.input`.
        """
        self.config = config
        self.device = device
        self.model = HifiganGenerator()
        hifigan_ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/bshall/hifigan/releases/download/v0.1/hifigan-hubert-discrete-bbad3043.pt",
            map_location="cpu",
        )
        consume_prefix_in_state_dict_if_present(hifigan_ckpt, "module.")
        self.model.load_state_dict(hifigan_ckpt)
        self.model.to(device)
        self.model.eval()
        self.model.remove_weight_norm()

    def run(self, batch):
        """
        Given the spectrogram, placed in the batch under the key `self.input`,
        computes and returns the spectrogram.
        """
        audio = batch[self.config.input.spectrogram]
        n_samples = batch[self.config.input.n_samples]
        n_samples *= self.config.upsampling_ratio
        audio_anon = self.model.forward(audio)
        return audio_anon, n_samples

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
