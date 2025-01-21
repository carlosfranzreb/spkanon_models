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

    def run(self, batch: dict) -> Tensor:
        """
        The batch contains a batch of audio samples as a tensor, and the string
        identifiers of the targets that should be used to convert the audio samples.
        """
        audio = batch["audio"].to(self.device)
        target = batch["target"]
        # TODO: compute ouput n_samples
        return self.model.convert(audio, target)

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
