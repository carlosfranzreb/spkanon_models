import importlib

from omegaconf import DictConfig
from nemo.collections.tts.models import FastPitchModel
import torch
from torch import Tensor
import torch.nn.functional as F


class FastPitch:
    def __init__(self, config: DictConfig, device: str):
        """
        The config must indicate under which key are placed the transcripts in the
        batch, under `config.input`. It may also indicate which speaker to use, under
        `config.speaker`. The model is loaded from the path specified in `config.init`.
        """

        self.config = config
        self.device = device
        self.n_targets = config.n_targets
        self.target_is_male = torch.zeros(
            self.n_targets, dtype=torch.bool, device=device
        )
        self.target_is_male[self.n_targets // 2 :] = True

        model_path = config.init
        if model_path.endswith(".nemo"):
            self.model = FastPitchModel.restore_from(restore_path=model_path)
        elif model_path.endswith(".ckpt"):
            self.model = FastPitchModel.load_from_checkpoint(checkpoint_path=model_path)
        else:
            self.model = FastPitchModel.from_pretrained(model_name=model_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.target_selection = None  # initialized later (see init_target_selection)

    def init_target_selection(self, cfg: DictConfig, *args):
        """
        Initialize the target selection algorithm. This method is called by the
        anonymizer, passing it config and the arguments that the defined algorithm
        requires. These are passed directly to the algorithm, along with the indices
        representing the speakers.
        """

        targets = torch.arange(self.n_targets).to(self.device)
        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(targets, cfg, self.target_is_male, *args)

    def run(self, batch: list) -> dict:
        """
        The input `batch` is a dict with a key `self.input` that contains the
        transcripts. For each transcript, parse it with the model's parser (performs
        text normalization and character tokenization) and then generate a spectrogram
        with those tokens. Return the spectrograms as a tensor.
        """

        # get the texts and the source speakers
        texts = batch[self.config.input.text]
        source = batch[self.config.input.source]
        source_is_male = batch[self.config.input.source_is_male].to(self.device)

        # pass a dummy input to the target selection algorithm
        n_tokens = torch.zeros(len(texts), dtype=torch.int64, device=self.device)
        target = self.target_selection.select(n_tokens, source, source_is_male)

        # compute the tokens from the transcripts
        tokens = list()
        for text_idx, text in enumerate(texts):
            tokens.append(self.model.parse(text).squeeze())
            n_tokens[text_idx] = tokens[-1].shape[0]
        tokens = _pad_tensors(tokens)

        (
            spect,
            dec_lens,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
            energy_pred,
            energy_tgt,
        ) = self.model.fastpitch(
            text=tokens,
            durs=None,
            pitch=None,
            speaker=target,
            pace=1.0,
            reference_spec=None,
            reference_spec_lens=None,
        )

        return {"spectrogram": spect, "lengths": dec_lens, "target": target}

    def to(self, device: str):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.model.to(device)
        self.target_selection.target_is_male = self.target_selection.target_is_male.to(
            device
        )
        self.device = device


def _pad_tensors(lists: list) -> Tensor:
    """`lists` is a list of tensors. Pad them so they have the same length and
    make a tensor with them."""
    max_values = max([value.shape[0] for value in lists])
    return torch.cat(
        [
            F.pad(value, (0, max_values - value.shape[0]), value=0).unsqueeze(0)
            for value in lists
        ]
    )
