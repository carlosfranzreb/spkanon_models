from omegaconf import DictConfig
import torch
import torch.nn.functional as F

from nemo.collections.tts.models import HifiGanModel


class HifiGan:
    def __init__(self, config: DictConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        model_init = config.init
        if model_init.endswith(".yaml"):
            self.model = HifiGanModel(config)
        elif model_init.endswith(".nemo"):
            self.model = HifiGanModel.restore_from(restore_path=model_init)
        elif model_init.endswith(".ckpt"):
            self.model = HifiGanModel.load_from_checkpoint(checkpoint_path=model_init)
        else:
            self.model = HifiGanModel.from_pretrained(model_name=model_init)
        del self.model.mpd
        del self.model.msd
        self.model.eval()
        self.model = self.model.to(device)

        # compute the upsample factor of the model
        upsample_rates = self.model.cfg["generator"]["upsample_rates"]
        self.upsample_factor = int(torch.prod(torch.tensor(upsample_rates)).item())

    def run(self, batch: list) -> tuple:
        self.x = batch[self.config.input.spectrogram]
        lens = batch[self.config.input.lengths]
        n_samples = lens * self.upsample_factor
        
        m = self.model.generator
        self.x = m.conv_pre(self.x)
        for upsample_layer, resblock_group in zip(m.ups, m.resblocks):
            self.x = F.leaky_relu(self.x, m.lrelu_slope)
            self.x = upsample_layer(self.x)
            self.xs = torch.zeros_like(self.x)
            for resblock in resblock_group:
                self.tmp = resblock(self.x).detach()
                self.xs = self.xs + self.tmp
            self.x = self.xs / m.num_kernels
            
        self.x = F.leaky_relu(self.x)
        self.x = m.conv_post(self.x)
        self.x = torch.tanh(self.x)

        out = self.x.clone()
        self.reset()

        return out, n_samples

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