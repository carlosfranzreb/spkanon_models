"""
Acoustic model for the SoftVC model. It is preceded by a HuBERT model and followed
by a HiFiGAN model.
"""

import os
import json
import importlib

import torch
from omegaconf import DictConfig

from spkanon_eval.component_definitions import InferComponent

# these targets are not available in the model
EXCLUDE = [1723, 328, 445, 441]


class Selector(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.target_selection = None

        # gather the identifiers and genders of the target speakers
        target_df = os.path.join(config.exp_folder, "data", "targets.txt")
        self.target_labels = list()
        self.target_is_male = list()
        with open(target_df, "r") as f:
            for line in f:
                obj = json.loads(line.strip())
                spkid = int(obj["speaker_id"])
                while len(self.target_labels) <= spkid:
                    self.target_labels.append(None)
                    self.target_is_male.append(None)
                if self.target_labels[spkid] is None:
                    self.target_labels[spkid] = int(obj["label"])
                    self.target_is_male[spkid] = obj["gender"] == "M"

        # ensure that the target datafile is LibriSpeech-train-clean-100
        assert len(self.target_labels) == 251 and 1034 in self.target_labels

        # remove the excluded targets
        for label in EXCLUDE:
            index = self.target_labels.index(label)
            del self.target_labels[index]
            del self.target_is_male[index]

        self.target_labels = torch.tensor(self.target_labels)
        self.target_is_male = torch.tensor(self.target_is_male)

    def init_target_selection(self, cfg: DictConfig, *args):
        """
        Initialize the target selection algorithm. This method is called by the
        anonymizer, passing it config and the arguments that the defined algorithm
        requires. These are passed directly to the algorithm, along with the target
        features computed in the constructor.
        """
        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(
            self.target_labels, cfg, target_is_male=self.target_is_male, *args
        )

    def run(self, batch: dict) -> dict:
        """
        Return the target labels.
        """
        audio = batch[self.config.input.audio].to("cpu")
        source = batch[self.config.input.source].to("cpu")
        source_is_male = batch[self.config.input.source_is_male].to("cpu")
        target = self.target_selection.select(audio, source, source_is_male)
        target = torch.tensor([self.target_labels[t] for t in target])
        return {"target": target}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
