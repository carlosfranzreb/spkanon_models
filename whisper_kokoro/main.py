import os
import json

from omegaconf import DictConfig
import torch
from torch.nn.utils.rnn import pad_sequence
from kokoro import KPipeline
import nltk


class KokoroWrapper:
    def __init__(self, config: DictConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.pipeline = KPipeline(lang_code=config.lang_code)
        self.upsampling_rate = 1800
        nltk.download("punkt_tab")

        # load target speakers
        target_df = os.path.join(config.exp_folder, "data", "targets.txt")
        self.targets = dict()
        for line in open(target_df):
            obj = json.loads(line)
            self.targets[obj["speaker_id"]] = obj["label"]

        self.targets = [self.targets[idx] for idx in range(len(self.targets))]
        self.targets = [
            self.pipeline.load_voice(tgt).squeeze(1) for tgt in self.targets
        ]
        self.targets = torch.stack(self.targets).to(device)

        self.target_selection = None  # initialized by Anonymizer

    def run(self, batch: list) -> tuple:

        # get the texts and the target speakers
        texts = batch[self.config.input.text]
        targets = self.target_selection.select(batch)

        # split texts into sentences and keep track of texts
        sentences, sentence2text = list(), list()
        sentence_targets = list()
        for idx, text in enumerate(texts):
            text_sentences = nltk.tokenize.sent_tokenize(
                text, language=self.config.tokenizer_language
            )
            sentences.extend(text_sentences)
            sentence2text.extend([idx] * len(text_sentences))
            sentence_targets.extend([targets[idx].item()] * len(text_sentences))

        sentence_targets = torch.tensor(sentence_targets)

        # phonemize sentences
        tuple_idx = 1 if self.config.lang_code in "ab" else 0
        tokens = [self.pipeline.g2p(text)[tuple_idx] for text in sentences]

        if self.config.lang_code in "ab":
            phones = list()
            for token in tokens:
                for gs, ps, tks in self.pipeline.en_tokenize(token):
                    if not ps:
                        continue

                    if len(ps) > 510:
                        ps = ps[:510]

                phones.append(ps)
        else:
            phones = tokens

        # define target voices
        voice_indices = [len(ps) - 1 for ps in phones]
        voices = self.targets[sentence_targets, voice_indices]

        # tokenize phones
        context_len = self.pipeline.model.context_length
        input_ids = list()
        for phone in phones:
            input_id = list(
                filter(
                    lambda i: i is not None,
                    map(lambda p: self.pipeline.model.vocab.get(p), phone),
                )
            )
            assert len(input_id) + 2 <= context_len, (len(input_id) + 2, context_len)
            input_ids.append(torch.tensor([0, *input_id, 0], device=self.device))

        input_lengths = torch.tensor([len(input_id) for input_id in input_ids])
        input_ids = pad_sequence(input_ids, batch_first=True)

        # create mask for the batch
        text_mask = (
            torch.arange(input_lengths.max())
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)

        # predict duration
        m = self.pipeline.model
        bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        d = m.predictor.text_encoder(d_en, voices[:, 128:], input_lengths, text_mask)
        x, _ = m.predictor.lstm(d)
        duration = m.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / self.config.speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(1)

        # apply durations to indices
        indices = [
            torch.repeat_interleave(
                torch.arange(input_ids.shape[1], device=m.device), pred_dur[idx]
            )
            for idx in range(pred_dur.shape[0])
        ]
        indices = torch.nn.utils.rnn.pad_sequence(indices)

        # one-hot alignment between input tokens and output frames
        pred_aln_trg = torch.zeros(
            (*input_ids.shape, indices.shape[0]), device=m.device
        )
        indices_range = torch.arange(indices.shape[0])
        for idx in range(pred_aln_trg.shape[0]):
            pred_aln_trg[idx, indices[:, idx], indices_range] = 1

        # f0 and N prediction
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = m.predictor.F0Ntrain(en, voices[:, 128:])

        # decode
        t_en = m.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        with torch.no_grad():
            audios = m.decoder(asr, F0_pred, N_pred, voices[:, :128])

        # merge audios of sentences to form texts again
        audios = audios.squeeze(1)
        full_audios = [torch.tensor([])] * len(texts)
        full_lengths = torch.zeros(len(text), dtype=torch.int)
        for sentence_idx, text_idx in enumerate(sentence2text):
            full_audios[text_idx] = torch.hstack(
                [full_audios[text_idx], audios[sentence_idx]]
            )
            full_lengths[text_idx] += input_lengths[sentence_idx]

        full_audios = pad_sequence(full_audios, batch_first=True)
        full_audios = full_audios.unsqueeze(1)

        # compute audio_lens and return
        audio_lens = full_lengths * self.upsampling_rate

        return full_audios, audio_lens, targets

    def to(self, device: str):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)
