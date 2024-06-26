import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig
from TTS.tts.layers.bark.inference_funcs import codec_decode

from spkanon_eval.component_definitions import InferComponent
from .bark_model import SingletonBarkVC


class EncodecDecoder(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        """
        The model is loaded following the instructions in the notebook provided in
        the repository https://github.com/bshall/knn-vc.
        - The config must indicate under which key are placed the WavLM matrices in the
            batch, under `config.input`.
        """
        self.device = device
        self.config = config
        self.model = SingletonBarkVC().bark
        self.model.to(self.device)

    @torch.inference_mode()
    def run(self, batch: list) -> tuple[Tensor, Tensor]:
        """
        Given the spectrogram, placed in the batch under the key `self.input`,
        computes and returns the spectrogram.
        """
        n_samples = (
            torch.ones(batch[self.config.input.n_feats].shape[0], dtype=torch.int32)
            * -1
        )
        waves = list()

        # TODO: batchify this
        data = batch[self.config.input.encodec]
        lens = batch[self.config.input.n_feats]
        for idx in range(data.shape[0]):
            codes = batch[self.config.input.encodec][idx, : lens[idx]]
            decoded = codec_decode(codes.T.cpu().numpy(), self.model)
            waves.append(torch.from_numpy(decoded))
            n_samples[idx] = waves[-1].shape[0]

        waves = pad_sequence(waves, batch_first=True).unsqueeze(1)
        return waves, n_samples

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
