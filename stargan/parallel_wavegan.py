"""
Wrapper for the Parallel WaveGAN model (https://pypi.org/project/parallel-wavegan/)
"""


from parallel_wavegan.utils import load_model
import torch
from torch import Tensor
from omegaconf import DictConfig

from spkanon_eval.component_definitions import InferComponent


class ParallelWaveGAN(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        self.device = device
        self.config = config
        self.model = load_model(config.init).to(device)
        self.model.remove_weight_norm()
        self.model.eval()
        self.replication_pad = torch.nn.ReplicationPad1d(self.model.aux_context_window)
        self.upsample_factor = self.model.upsample_factor.item()

    def run(self, batch: list) -> tuple[Tensor, Tensor]:
        """
        Run the model on the given batch.
        Input dims: (batch_size, n_mels, n_frames)
        Output dims: (batch_size, n_samples)
        """
        spec = batch[self.config.input.spectrogram]
        input_len = batch[self.config.input.n_frames]
        x = torch.randn(
            spec.shape[0], 1, spec.shape[2] * self.upsample_factor
        ).to(self.device)
        c = self.replication_pad(spec)
        out = self.model.forward(x, c)
        n_samples = input_len * self.upsample_factor
        return out, n_samples

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)
