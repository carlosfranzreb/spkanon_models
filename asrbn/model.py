import torch
from torch import Tensor
from omegaconf import DictConfig

from spkanon_eval.component_definitions import InferComponent


class ASRBN(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        """
        The model is loaded following the instructions in the notebook provided in
        the repository https://github.com/bshall/soft-vc.
        - The config must indicate under which key are placed the spectrograms in the
            batch, under `config.input`.
        """
        self.config = config
        self.device = device
        self.model = torch.hub.load(
            "deep-privacy/SA-toolkit",
            "anonymization",
            tag_version=config.ckpt,
            trust_repo=True,
        )
        self.model.to(device)
        self.model.eval()

    def run(self, batch: dict) -> tuple[Tensor, Tensor]:
        """
        The batch contains a batch of audio samples as a tensor, and the string
        identifiers of the targets that should be used to convert the audio samples.
        """
        audio = batch["audio"].to(self.device)
        n_samples = batch["n_samples"]
        target = [str(t.item()) for t in batch["target"]]
        audio_anon = self.model.convert(audio, target)
        return audio_anon, n_samples

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
