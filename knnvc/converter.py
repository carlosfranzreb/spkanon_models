"""
Selects targets based on speaker embeddings.
"""

import os
import json
import logging
import importlib

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig

from spkanon_eval.setup_module import setup as setup_module
from spkanon_eval.datamodules.dataloader import eval_dataloader
from spkanon_eval.component_definitions import InferComponent


LOGGER = logging.getLogger("progress")


class Converter(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        """
        Initialize the converter.

        It requires a datafile for the targets in the path data/targets.txt under
        the `exp_folder`, which is present in the config object.

        You can also pass a directory with the WavLM features of the targets
        (config.target_feats), as dumped by a previous run. If this directory is not
        given, the features are extracted here. They are also dumped, so that they can
        be reused in future runs.

        Args:
            config: we assume that all samples of each speaker are consecutive.
            device: the device where the model should run.
        """
        self.config = config
        self.device = device
        self.target_selection = None  # initialized later (see init_target_selection)
        self.target_feats = list()
        self.target_is_male = list()
        target_df = os.path.join(config.exp_folder, "data", "targets.txt")

        # if possible, load the WavLM features of the targets
        target_feats_dir = config.get("target_feats", None)
        if target_feats_dir is not None:
            LOGGER.info(f"Loading target features from {target_feats_dir}")
            for spk_idx in range(len(os.listdir(target_feats_dir))):
                self.target_feats.append(
                    torch.load(
                        os.path.join(target_feats_dir, f"{spk_idx}.pt"),
                        map_location="cpu",
                    )
                )

            # gather the genders of the target speakers
            LOGGER.info("Loading the genders of the target speakers")
            self.target_is_male = torch.zeros(len(self.target_feats), dtype=torch.bool)
            with open(target_df, "r") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    spk = obj["speaker_id"]
                    self.target_is_male[spk] = obj["gender"] == "M"

            return

        # otherwise compute the WavLM features and concatenate them for each speaker
        LOGGER.info("Extracting target features")
        wavlm = setup_module(config.wavlm, device)
        dl = eval_dataloader(config.wavlm_dl, target_df, wavlm)
        for _, batch, data in dl:
            feats, feat_lengths = wavlm.run(batch).values()
            for idx in range(len(feats)):
                spk = data[idx]["speaker_id"]
                while len(self.target_feats) <= spk:
                    self.target_feats.append(None)
                    self.target_is_male.append(None)

                self.target_is_male[spk] = data[idx]["gender"] == "M"
                unpadded_feats = feats[idx, : feat_lengths[idx]].to("cpu")
                if self.target_feats[spk] is None:
                    self.target_feats[spk] = unpadded_feats
                else:
                    self.target_feats[spk] = torch.cat(
                        [self.target_feats[spk], unpadded_feats], dim=0
                    )

        self.target_is_male = torch.tensor(self.target_is_male)

        # dump the features
        dump_folder = os.path.join(config.exp_folder, "target_feats")
        os.makedirs(dump_folder)
        for idx in range(len(self.target_feats)):
            torch.save(self.target_feats[idx], os.path.join(dump_folder, f"{idx}.pt"))

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
            self.target_feats, cfg, target_is_male=self.target_is_male, *args
        )

    def run(self, batch: list) -> dict:
        """
        Selects the target speakers for the given batch if needed and converts the batch
        to those targets.

        Args:
            batch: a list with a tensor comprising spectrograms in first position.

        TODO: Check whether running this on GPU performs better.
        """
        # get the features, source and target speakers
        feats = batch[self.config.input.feats].to("cpu")
        n_feats = batch[self.config.input.n_feats].to("cpu")
        source = batch[self.config.input.source].to("cpu")
        source_is_male = batch[self.config.input.source_is_male].to("cpu")
        target = self.target_selection.select(feats, source, source_is_male)

        # run the conversion
        converted_feats = list()
        for idx, target_idx in enumerate(target):
            source_feats = feats[idx, : n_feats[idx]]
            converted_feats.append(
                self.convert_vecs(source_feats, self.target_feats[target_idx])
            )
        converted_batch = pad_sequence(converted_feats, batch_first=True)

        return {
            "wavlm": converted_batch.to(self.device),
            "target": target,
            "n_feats": n_feats,
        }

    def convert_vecs(self, source_vecs: Tensor, target_vecs: Tensor) -> Tensor:
        """
        Given the WavLM vecs of the source and target audios, convert them with the
        KnnVC matching algorithm.

        Args:
            source_vec: tensor of shape (n_vecs_s, vec_dim)
            target_vecs: tensor of shape (n_vecs_t, vec_dim)

        Returns:
            converted wavLM vectors: tensor of shape (n_vecs_s, vec_dim)
        """
        cos_sim = cosine_similarity(source_vecs, target_vecs)
        best = cos_sim.topk(k=self.config.n_neighbors, dim=1)
        return target_vecs[best.indices].mean(dim=1)

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.target_selection.target_is_male = self.target_selection.target_is_male.to(
            device
        )


def cosine_similarity(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    """
    Compute the cosine similarity among all vectors in `tensor_a` and `tensor_b`.

    Args:
        tensor_a: tensor of shape (n_vecs_a, vec_dim)
        tensor_b: tensor of shape (n_vecs_b, vec_dim)

    Returns:
        cosine similarity tensor: tensor of shape (n_vecs_a, n_vecs_b)
    """
    dot_product = torch.matmul(tensor_a, tensor_b.transpose(-1, -2))
    source_norm = torch.norm(tensor_a, dim=-1)
    target_norm = torch.norm(tensor_b, dim=-1)
    cos_sim = dot_product / torch.outer(source_norm, target_norm)
    return cos_sim
