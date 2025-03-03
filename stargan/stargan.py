"""
Wrapper for the StarGANv2-VC model (https://github.com/yl4579/StarGANv2-VC).
This implementation follows the one provided in the demo.

There are two kinds of conversion:
- With reference: the reference audiofile is used to compute the speaker style vector
- Without reference: the speaker style vector is computed with the mapping network

Currently, only the first kind is implemented.
"""

import importlib

from omegaconf import OmegaConf, DictConfig
import torch
from torch import Tensor

from StarGANv2VC.models import (
    Generator,
    MappingNetwork,
)
from StarGANv2VC.Utils.JDC.model import JDCNet

from spkanon_eval.component_definitions import InferComponent


SAMPLE_RATE = 24000  # model's sample rate
TARGET_IS_MALE = {
    225: False,
    228: False,
    229: False,
    230: False,
    231: False,
    233: False,
    236: False,
    239: False,
    240: False,
    244: False,
    226: True,
    227: True,
    232: True,
    243: True,
    254: True,
    256: True,
    258: True,
    259: True,
    270: True,
    273: True,
}


class StarGAN(InferComponent):
    def __init__(self, config: DictConfig, device: str) -> None:
        """
        The config must indicate under which key are placed the transcripts in the
        batch, under `config.input`. It may also indicate which speaker to use, under
        `config.speaker`.

        The model is loaded from the path specified in `config.init` The training
        script offered in the repo stores its files as `.pth` files.

        Once the model is loaded, we can already compute the speaker style vector,
        which will be used to convert all input samples.
        """

        self.config = config
        self.device = device

        if not config.init.endswith(".pth"):
            raise ValueError("The model checkpoint must be a .pth file.")
        self.generator, self.mapping_network, self.f0_model = init_model(config, device)

        self.target_selection = None  # initialized later (see init_target_selection)

    def init_target_selection(self, cfg: DictConfig, *args):
        """
        Initialize the target selection algorithm. This method is called by the
        anonymizer, passing it config and the arguments that the defined algorithm
        requires. These are passed directly to the algorithm, along with the style
        vectors of the StarGAN.
        """

        style_vecs = self.mapping_network(
            torch.randn(
                self.config.n_targets, self.mapping_network.shared[0].in_features
            ).to(self.device),
            torch.arange(self.config.n_targets).to(self.device),
        )

        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        target_is_male = torch.tensor(list(TARGET_IS_MALE.values())).to(self.device)
        self.target_selection = cls(style_vecs, cfg, target_is_male, *args)

    def run(self, batch: list) -> dict[str, Tensor]:
        """
        Convert the input spectrogram to the target speaker style.
        Input and output dims: (batch_size, n_mels, n_frames)
        If a target is given as part of the batch, it is used to compute the style
        style vector. Otherwise, the target selection algorithm is used to select
        the target speaker. Targets may be -1, in which case the target selection
        algorithm is used. If speaker consistency is enabled, the target speakers
        must be consistent across the utterances of each speaker. This is checked
        here and may overwrite the given target speaker.
        """

        # get the spectrograms and target speakers
        spec = batch[self.config.input.spectrogram]
        source = batch[self.config.input.source]
        source_is_male = batch[self.config.input.source_is_male]
        target = self.target_selection.select(spec, source, source_is_male)

        # compute the speaker style vector
        latent_dim = self.mapping_network.shared[0].in_features
        style_vec = self.mapping_network(
            torch.randn(target.shape[0], latent_dim).to(self.device), target
        )

        # run the StarGAN model
        spec_in = self._normalize_spec(spec).unsqueeze(1)
        f0_feats = self.f0_model.get_feature_GAN(spec_in)
        spec_out = self.generator(spec_in, style_vec, F0=f0_feats)

        return {
            "spectrogram": spec_out.squeeze(1),
            "n_frames": batch[self.config.input.n_frames],
            "target": target,
        }

    def _normalize_spec(self, spec: Tensor) -> Tensor:
        """Normalize a spectrogram with mean=-4 and std=4"""
        return (torch.log(1e-5 + spec) + 4) / 4

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.generator.to(self.device)
        self.mapping_network.to(self.device)
        self.f0_model.to(self.device)
        self.target_selection.target_is_male = self.target_selection.target_is_male.to(
            device
        )

def init_model(config, device):
    """
    Initialize the generator, the mapping network and the F0 model.
    Store them as attributes and load the weights stored in the checkpoint.
    Its path is stored in the config under `config.init`.
    """
    model_cfg = OmegaConf.load(config.config).model_params
    # load models in evaluation mode
    generator = Generator(
        model_cfg.dim_in,
        model_cfg.style_dim,
        model_cfg.max_conv_dim,
        w_hpf=model_cfg.w_hpf,
        F0_channel=model_cfg.F0_channel,
    )
    mapping_network = MappingNetwork(
        model_cfg.latent_dim,
        model_cfg.style_dim,
        model_cfg.num_domains,
        hidden_dim=model_cfg.max_conv_dim,
    )
    # load the weights
    weights = torch.load(config.init, map_location="cpu")["model_ema"]
    generator.load_state_dict(weights["generator"])
    mapping_network.load_state_dict(weights["mapping_network"])
    f0_model = JDCNet(num_class=1, seq_len=192)
    f0_model.load_state_dict(torch.load(config.f0_ckpt, map_location="cpu")["net"])
    # move models to the device
    generator.to(device)
    mapping_network.to(device)
    f0_model.to(device)
    # run models in evaluation mode
    generator.eval()
    mapping_network.eval()
    f0_model.eval()
    return generator, mapping_network, f0_model
