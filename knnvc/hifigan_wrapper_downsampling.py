import json

import torch
import torchaudio
from torch import Tensor
from omegaconf import DictConfig

from spkanon_models.knnvc.hifigan_model import Generator, AttrDict
from spkanon_eval.component_definitions import InferComponent


class Hifigan(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        """
        The model is loaded following the instructions in the notebook provided in
        the repository https://github.com/bshall/knn-vc.
        - The config must indicate under which key are placed the WavLM matrices in the
            batch, under `config.input`.
        """
        self.config = config
        self.device = device
        self.model = Generator(AttrDict(json.load(open(config.config))))
        ckpt = config.get("ckpt", None)
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location=device)
            self.model.load_state_dict(state_dict["generator"])
        self.model.to(device)
        self.model.eval()
        self.model.remove_weight_norm()

        downsample_to = config.synthesizer_params.get("downsample_to", None)
        self.downsample = None
        if downsample_to is not None:
            self.downsample = torchaudio.transforms.Resample(
                config.sample_rate, downsample_to
            ).to(device)
            self.upsample = torchaudio.transforms.Resample(
                downsample_to, config.sample_rate
            ).to(device)

    @torch.inference_mode()
    def run(self, batch: list) -> tuple[Tensor, Tensor]:
        """
        Given the spectrogram, placed in the batch under the key `self.input`,
        computes and returns the spectrogram.
        """
        n_samples = batch[self.config.input.n_feats] * self.config.hop_length
        waves = self.model.forward(batch[self.config.input.wavlm])

        if self.downsample is not None:
            waves = self.upsample(self.downsample(waves))
            waves -= waves.mean(dim=0)
            waves /= waves.std(dim=0)

        return waves, n_samples

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
