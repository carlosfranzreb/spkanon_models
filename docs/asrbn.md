# Soft-VC

ASRBN is baseline B5 from the VoicePrivacy Challenge 2024. It was developed by Pierre Champion as part of his thesis. It is also summarised in the VPC24 evaluation plan.

- Pierre's thesis: <https://arxiv.org/abs/2308.04455>
- VPC24 evaluation plan: <https://www.voiceprivacychallenge.org/vp2024/docs/VoicePrivacy_2024_Eval_Plan_v2.1.pdf#page=12.26>
- GitHub repository: <https://github.com/deep-privacy/SA-toolkit/>

## Installation

This model is downloaded automatically with `torch.hub`. It does not require any additional packages.

## Implementation

Our implementation uses the whole model at once through torch.hub. See `asrbn/model.py`. The target selection is done in `asrbn/featproc.py`. There are 247 available targets from Librispeech's train-clean-100 subset (all but four speakers).
