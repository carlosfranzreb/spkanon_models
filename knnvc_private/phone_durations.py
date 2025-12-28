"""
Feature extractor that computes phone durations, based on:

Tomashenko, Natalia, et al. “Exploiting Context-Dependent Duration Features for Voice
Anonymization Attack Systems.” Interspeech 2025.

This feature extractor can be used to replace the spectrogram when training the speaker
recognizer, to assess how much speaker identity is being leaked through phone durations
alone, without any additional speaker information.

! It requires installing private kNN-VC through the spkanon_models repository:
<https://github.com/carlosfranzreb/spkanon_models>

To use it, the spkid model should be trained with config
`spane/config/components/asv/spkid/train_ecapa_phone_durations.yaml`
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from speechbrain.utils.autocast import fwd_default_precision
from omegaconf import OmegaConf

from spkanon_eval.setup_module import setup as setup_module
from spkanon_models.knnvc_private.conv_decoder import load_model as load_conv_decoder


class PhoneDurations(torch.nn.Module):
    """
    Extract phone durations from utterances, with the phone recognizer from private
    kNN-VC.

    Args:
        device: cpu or cuda
        phone_lexicon: text file with one phone per line. It should be the same lexicon
            the predictor was trained on.
        wavlm_ckpt: checkpoint for wavlm
        phone_predictor_ckpt: checkpoint for private kNN-VC's phone decoder.
        ignore_durs: if true, durations are ignored: all durations are set to 1.
    """

    def __init__(
        self,
        device: str,
        phone_lexicon: str,
        wavlm_ckpt: str,
        phone_predictor_ckpt: str,
        ignore_durs: bool,
    ):
        super().__init__()
        self.device = device
        self.phone_lexicon = open(phone_lexicon, "r").read().splitlines()
        self.phone_predictor = load_conv_decoder(phone_predictor_ckpt, device)

        wavlm_cfg = OmegaConf.create(
            {
                "cls": "spkanon_eval.featex.wavlm.wrapper.WavlmWrapper",
                "ckpt": wavlm_ckpt,
                "layer": 24,
                "hop_length": 320,
            }
        )
        self.wavlm = setup_module(wavlm_cfg, device)
        self.ignore_durs = ignore_durs

    @torch.inference_mode()
    @fwd_default_precision(cast_inputs=torch.float32)
    def forward(self, wav: Tensor) -> Tensor:
        """
        Returns a set of features generated from the input waveforms.

        Args:
            wav: batch of audio signals to transform to features.

        Returns:
            feats: sequence of one-hot vectors of the size of the number of phones.
                Instead of a 1, the number in the vector is the phone's duration.
        """
        batch = [wav, None, torch.ones(wav.shape[0], dtype=torch.int) * wav.shape[1]]
        feats, feat_lengths = self.wavlm.run(batch).values()
        phones = self.phone_predictor(feats).argmax(dim=2)

        # compute the phone durations
        feats = list()
        for utt_idx in range(phones.shape[0]):
            unique_phones, phone_durations = torch.unique_consecutive(
                phones[utt_idx], return_counts=True
            )
            utt_feats = torch.zeros(
                (unique_phones.shape[0], len(self.phone_lexicon)),
                dtype=torch.long,
                device=self.device,
            )

            if not self.ignore_durs:
                utt_feats[torch.arange(utt_feats.shape[0]), unique_phones] = (
                    phone_durations
                )
            else:
                utt_feats[torch.arange(utt_feats.shape[0]), unique_phones] = 1

            feats.append(utt_feats)

        return pad_sequence(feats, batch_first=True).to(torch.float)


class TestPhoneDurationsForward(unittest.TestCase):
    def setUp(self):
        # create a temporary lexicon
        self.lexicon_file = "test_lexicon.txt"
        with open(self.lexicon_file, "w") as f:
            f.write("\n".join(["AA", "BB", "CC"]))

        # patch model loading and external calls
        self.mock_predictor = MagicMock()
        self.mock_wavlm = MagicMock()
        patcher1 = patch(
            __name__ + ".load_conv_decoder", return_value=self.mock_predictor
        )
        patcher2 = patch(__name__ + ".setup_module", return_value=self.mock_wavlm)
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.mock_load_conv = patcher1.start()
        self.mock_setup_mod = patcher2.start()

        # instantiate PhoneDurations
        self.model = PhoneDurations(
            device="cpu",
            phone_lexicon=self.lexicon_file,
            wavlm_ckpt="fake_wavlm",
            phone_predictor_ckpt="fake_ckpt",
            ignore_durs=False,
        )

    def test_forward_basic_durations(self):
        """Test that repeated phones produce correct durations."""
        wav = torch.randn(1, 10)  # fake audio batch

        # Mock wavlm output
        feats = torch.randn(1, 10, 5)
        self.mock_wavlm.run.return_value = {
            "feats": feats,
            "feat_lengths": torch.tensor([10]),
        }

        phones = torch.tensor([[0, 0, 1, 1, 1, 2]])
        self.mock_predictor.return_value = torch.nn.functional.one_hot(
            phones, num_classes=3
        ).float()

        # expect 3 rows (phones 0,1,2) with durations [2,3,1]
        self.mock_predictor.return_value.argmax = lambda dim: phones
        out = self.model.forward(wav)
        expected = torch.tensor([[2, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=torch.float)
        self.assertTrue(torch.equal(out[0], expected))

    def test_forward_ignore_durations(self):
        """Test that ignore_durs=True sets all durations to 1."""
        self.model.ignore_durs = True
        wav = torch.randn(1, 6)

        self.mock_wavlm.run.return_value = {
            "feats": torch.randn(1, 6, 5),
            "feat_lengths": torch.tensor([6]),
        }
        phones = torch.tensor([[0, 0, 1, 1, 2, 2]])
        self.mock_predictor.return_value = torch.nn.functional.one_hot(
            phones, num_classes=3
        ).float()
        self.mock_predictor.return_value.argmax = lambda dim: phones

        out = self.model.forward(wav)
        expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
        self.assertTrue(torch.equal(out[0], expected))

    def test_forward_padding_multiple_utts(self):
        """Test that different utterance lengths are padded correctly."""
        wav = torch.randn(2, 8)

        self.mock_wavlm.run.return_value = {
            "feats": torch.randn(2, 8, 5),
            "feat_lengths": torch.tensor([8, 8]),
        }
        phones = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 2, 2],  # longer
                [1, 1, 2, 2, 2, 2, 2, 2],  # shorter (fewer unique phones)
            ]
        )
        self.mock_predictor.return_value = torch.nn.functional.one_hot(
            phones, num_classes=3
        ).float()
        self.mock_predictor.return_value.argmax = lambda dim: phones

        out = self.model.forward(wav)

        # First utterance has 3 unique phones
        self.assertEqual(out.shape[1], 3)
        # Check that padding keeps shorter utterance aligned
        self.assertTrue((out[1, -1] == 0).all())  # last row should be padding


if __name__ == "__main__":
    unittest.main()
