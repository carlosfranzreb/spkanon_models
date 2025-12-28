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


class Selector(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.target_selection = None  # initialized by Anonymizer

    def run(self, batch: dict) -> dict:
        """
        Return the target labels.
        """
        audio = batch[self.config.input.audio].to("cpu")
        source = batch[self.config.input.source].to("cpu")
        source_is_male = batch[self.config.input.source_is_male].to("cpu")
        target = self.target_selection.select(audio, source, source_is_male)
        target_labels = torch.tensor([self.target_labels[t] for t in target])
        return {"target": target, "target_labels": target_labels}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
