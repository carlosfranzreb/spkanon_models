"""
HuBERT component of the Bark-VC model.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from TTS.tts.layers.bark.hubert.hubert_manager import HubertManager
from TTS.tts.layers.bark.hubert.kmeans_hubert import CustomHubert
from TTS.tts.layers.bark.hubert.tokenizer import HubertTokenizer

from spkanon_eval.component_definitions import InferComponent


SAMPLE_RATE = 24000  # model's sample rate


class Hubert(InferComponent):
    def __init__(self, config, device):
        self.device = device
        hubert_manager = HubertManager()
        hubert_manager.make_sure_tokenizer_installed(
            model_path=config.hubert_tokenizer_path,
        )
        self.model = CustomHubert(
            checkpoint_path=config.hubert_path,
        ).to(self.device)
        self.model.eval()

        self.tokenizer = HubertTokenizer.load_from_checkpoint(
            config.hubert_tokenizer_path,
            map_location=self.device,
        )

    def run(self, batch):
        """
        Returns the acoustic units for the given NeMo batch, which is a tuple where
        the audio batch is placed in the first position.
        """
        audio = batch[0].to(self.device)
        # TODO: batchify this
        semantic_tokens = list()
        n_samples = batch[2]
        n_tokens = torch.ones(audio.shape[0], dtype=torch.int32) * -1
        for idx in range(audio.shape[0]):
            audio_idx = audio[idx, : n_samples[idx]].unsqueeze(0)
            vectors = self.model.forward(audio_idx, input_sample_hz=SAMPLE_RATE)
            tokens = self.tokenizer.get_token(vectors)
            semantic_tokens.append(tokens)
            n_tokens[idx] = tokens.shape[0]

        semantic_tokens = pad_sequence(semantic_tokens, batch_first=True)
        return {"feats": semantic_tokens, "n_feats": n_tokens}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)
