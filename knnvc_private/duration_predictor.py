import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import g2p_en


class DurationPredictor(nn.Module):
    """
    Wrapper for the duration predictor model of FastSpeech2.
    """

    def __init__(self, phone_lexicon: list[str], device: str) -> None:
        super().__init__()
        self.g2p_lexicon = g2p_en.G2p().phonemes
        self.fs_models, self.cfg, self.task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        TTSHubInterface.update_cfg_with_data_cfg(self.cfg, self.task.data_cfg)
        self.generator = self.task.build_generator(self.fs_models, self.cfg)

        self.phone_lexicon = phone_lexicon
        self.device = device
        self.model = self.fs_models[0]
        self.model.eval()
        self.model.to(device)

    def forward(self, phones: Tensor, stress: str = "0") -> Tensor:
        """
        Predicts the duration of each of the phones. The duration is measured
        in number of WavLM frames. One frame is 20 ms.

        Args:
            phones: a list comprising lists of phones, one per sample in the batch.
            stress: the stress of the phones. It can be "0", "1", or "2".

        Returns:
            durations: the predicted durations, of shape (batch_size, len(phones),).
        """
        src_tokens, src_lengths = list(), list()
        for sample in phones:
            phone_list = [self.phone_lexicon[phone_idx] for phone_idx in sample]
            src_tokens_, src_lengths_ = self.encode_phones(phone_list, stress)
            src_tokens.append(src_tokens_.squeeze())
            src_lengths.append(src_lengths_)

        src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=1).to(
            self.device
        )
        src_lengths = torch.cat(src_lengths, dim=0).to(self.device)

        log_dur_out = self.model(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=None,
            incremental_state=None,
            target_lengths=None,
            speaker=None,
        )[3]
        dur_out = torch.clamp(torch.round(torch.exp(log_dur_out) - 1).long(), min=0)
        dur_out = dur_out[:, :-1] / 1.72
        return dur_out

    def encode_phones(
        self, phones: list[str], stress: str = "0"
    ) -> tuple[Tensor, Tensor]:
        """
        Receives a list of strings, which are the phones predicted from WavLM features
        by the phone predictor. It returns their encodings in FastSpeech2.

        Args:
            phones: the list of phones.
            stress: the stress of the phones. It can be "0", "1", or "2".

        Returns:
            src_tokens: the encoded phones, of shape (1, len(phones), vec_dim).
            src_lengths: the length of the phones, of shape (1, len(phones),).
        """
        g2p_phones = list()
        for phone_idx in range(len(phones)):
            phone = phones[phone_idx]
            if phone in ["SIL", "SPN"]:
                phone = "sp"
            elif phone not in self.g2p_lexicon:
                phone += stress
            g2p_phones.append(phone)

        g2p_phones = " ".join(g2p_phones)
        tokenized = TTSHubInterface.tokenize(
            g2p_phones, self.task.data_cfg.bpe_tokenizer
        )
        src_tokens = self.task.src_dict.encode_line(
            tokenized, add_if_not_exist=False
        ).view(1, -1)
        src_lengths = torch.Tensor([len(phones)]).long()
        return src_tokens, src_lengths
