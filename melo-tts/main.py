import importlib

from omegaconf import DictConfig
import torch
import torch.nn.functional as F

from kokoro import KPipeline


class KokoroWrapper:
    def __init__(self, config: DictConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.pipeline = KPipeline(lang_code=config.lang_code)
        self.upsampling_rate = 1800

        # get target speakers from config
        self.target_selection = None  # initialized later (see init_target_selection)
        self.targets = list()
        self.target_is_male = list()
        for gender in ["F", "M"]:
            for target in config.targets[gender]:
                self.targets.append(target)
                self.target_is_male.append(gender == "M")

        # load target speakers
        self.targets = [
            self.pipeline.load_voice(tgt).squeeze(1) for tgt in self.targets
        ]
        self.targets = torch.stack(self.targets).to(device)
        self.target_is_male = torch.tensor(self.target_is_male, device=device)

    def init_target_selection(self, cfg: DictConfig, *args):
        """
        Initialize the target selection algorithm. This method is called by the
        anonymizer, passing it config and the arguments that the defined algorithm
        requires. These are passed directly to the algorithm, along with the target
        speaker data from Kokoro.
        """

        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(self.targets, cfg, self.target_is_male, *args)

    def run(self, batch: list) -> tuple:

        # get the texts and the source speakers
        texts = batch[self.config.input.text]
        source = batch[self.config.input.source]
        source_is_male = batch[self.config.input.source_is_male].to(self.device)

        # pass a dummy input to the target selection algorithm
        dummy = torch.zeros(len(texts), dtype=torch.int64, device=self.device)
        targets = self.target_selection.select(dummy, source, source_is_male)
        del dummy

        # phonemize texts
        tokens = [self.pipeline.g2p(text)[1] for text in texts]
        phones = list()
        for token in tokens:
            for gs, ps, tks in self.pipeline.en_tokenize(token):
                if not ps:
                    continue
                if len(ps) > 510:
                    ps = ps[:510]
            phones.append(ps)

        # define target voices
        voice_indices = [len(ps) - 1 for ps in phones]
        voices = self.targets[targets, voice_indices]

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
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)

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

        # compute audio_lens and return
        audio_lens = input_lengths * self.upsampling_rate
        return audios, audio_lens, targets

    def to(self, device: str):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)

    def reset(self):
        """Delete intermediate tensors from the forward pass."""
        del self.tmp
        del self.xs
        del self.x
        torch.cuda.empty_cache()
