import torch
from torch import Tensor

N_PHONES = 41


class PhonePredictor(torch.nn.Module):
    """
    The SUPERB phonetic classifier. This model is only used for inference,
    initialized with the weights of the model trained in the SUPERB project.
    """

    def __init__(self, ckpt_path: str, device: str) -> None:
        super(PhonePredictor, self).__init__()

        # init model
        ckpt = torch.load(ckpt_path, map_location=device)["Downstream"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            ckpt, prefix="model."
        )

        input_dim = ckpt["in_linear.weight"].shape[1]
        hidden_size = ckpt["in_linear.weight"].shape[0]
        self.in_linear = torch.nn.Linear(input_dim, hidden_size)
        self.out_linear = torch.nn.Linear(hidden_size, N_PHONES)
        self.drop = torch.nn.Dropout(0)
        self.act_fn = torch.nn.functional.relu

        # load weights, set device and eval mode
        self.load_state_dict(ckpt)
        self.to(device)
        self.eval()

    @torch.inference_mode()
    def forward(self, features: Tensor) -> Tensor:
        hidden = self.in_linear(features)
        hidden = self.drop(hidden)
        hidden = self.act_fn(hidden)
        predicted = self.out_linear(hidden)
        return predicted
