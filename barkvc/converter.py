"""
Acoustic model for the SoftVC model. It is preceded by a HuBERT model and followed
by a HiFiGAN model.
"""

import json
import os
import logging
import importlib

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from TTS.tts.layers.bark.inference_funcs import (
    load_npz,
    generate_voice,
    generate_coarse,
    generate_fine,
)

from spkanon_eval.component_definitions import InferComponent
from .bark_model import SingletonBarkVC


LOGGER = logging.getLogger("progress")
SAMPLE_RATE = 24000  # model's sample rate


class BarkVC(InferComponent):
    def __init__(self, config, device):
        self.device = device
        self.config = config
        self.model = SingletonBarkVC().bark
        self.model.to(self.device)
        self.target_selection = None  # initialized later (see init_target_selection)

        # get the speaker IDs and paths between 6 and 10s of the targets
        LOGGER.info("Loading target data")
        self.target_files = list()
        self.target_speakers = list()
        current_spk = -1
        spk_sample = None
        target_df = os.path.join(config.exp_folder, "data", "targets.txt")
        for line in open(target_df):
            obj = json.loads(line)
            spk = obj["speaker_id"]
            if current_spk == -1:
                current_spk = spk

            if spk != current_spk:
                self.target_files.append(spk_sample)
                self.target_speakers.append(current_spk)
                spk_sample = None
                current_spk = spk

            if spk_sample is None and (
                self.config["target_min_dur"]
                <= obj["duration"]
                <= self.config["target_max_dur"]
            ):
                spk_sample = obj["path"]

        self.target_files.append(spk_sample)
        self.target_speakers.append(current_spk)

        # if possible, load the Encodec features of the targets
        target_feats_dir = config.get("target_feats", None)
        self.target_feats = list()
        if target_feats_dir is not None:
            LOGGER.info(f"Loading target features from {target_feats_dir}")
            for idx in range(len(self.target_speakers)):
                self.target_feats.append(
                    load_npz(os.path.join(target_feats_dir, f"{idx}.npz"))
                )
            return

        # otherwise compute the Encodec features: first file between 6 and 10s is used
        LOGGER.info("Extracting target features")
        dump_folder = os.path.join(config.exp_folder, "target_feats")
        os.makedirs(dump_folder)
        for t_idx, t_file in enumerate(self.target_files):
            if t_file is None:
                continue
            dump_path = os.path.join(dump_folder, f"{t_idx}.npz")
            generate_voice(t_file, self.model, dump_path)
            self.target_feats.append(load_npz(dump_path))

    def init_target_selection(self, cfg: OmegaConf, *args):
        """
        Initialize the target selection algorithm. This method is called by the
        anonymizer, passing it config and the arguments that the defined algorithm
        requires. These are passed directly to the algorithm, along with the target
        features computed in the constructor.
        """
        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(self.target_feats, cfg, *args)

    def run(self, batch):
        """
        Given the spectrogram, placed in the batch under the key `self.input`,
        computes and returns the spectrogram.
        """
        # get the features, source and target speakers
        feats = batch[self.config.input.feats]
        n_feats_in = batch[self.config.input.n_feats]
        source = batch[self.config.input.source]
        target_in = (
            batch[self.config.input.target]
            if self.config.input.target in batch
            else None
        )
        target = self.target_selection.select(feats, source, target_in)

        # generate the converted encodec features, conditioned on the target
        # TODO: batchify this
        converted_feats = list()
        feats = feats.cpu().numpy()
        n_feats_out = torch.ones(len(target), dtype=torch.int32) * -1
        for idx, target_idx in enumerate(target):
            source_feats = feats[idx, : n_feats_in[idx]]
            converted_feats.append(
                self.generate_codes(source_feats, self.target_feats[target_idx])
            )
            n_feats_out[idx] = converted_feats[-1].shape[0]
        converted_batch = pad_sequence(converted_feats, batch_first=True)

        return {"encodec": converted_batch, "target": target, "n_feats": n_feats_out}

    def generate_codes(
        self, hubert_feats: np.ndarray, target_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the encodec codes for the given Hubert codes, conditioned on the
        target features, which are also encodec features. It returns them as a tensor
        of shape (n_codes, n_codebooks).
        ! We return them in this shape to enable the padding of the batch. This model
        ! processes them the other way around (n_codebooks, n_codes).
        """
        x_coarse_gen = generate_coarse(
            hubert_feats,
            self.model,
            history_prompt=target_feats,
            temp=0.7,
            base=None,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            self.model,
            history_prompt=target_feats,
            temp=0.5,
            base=None,
        )
        return torch.from_numpy(x_fine_gen).T

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
