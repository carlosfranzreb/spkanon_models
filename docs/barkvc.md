# Bark-VC

Bark-VC adapts the VALL-E approach, which is a text-to-speech model, to perform voice conversion. Suno AI open sourced an implementation of VALL-E, called [Bark](https://github.com/suno-ai/bark/blob/main/model-card.md). It differs from VALL-E in that it represents the input text with HuBERT features instead of phonemes, making its adaptation for voice conversion a breeze: instead of computing the HuBERT features from a text, simply extract the HuBERT features from the input audio. The rest of the pipeline stays intact, allowing this approach to use the pre-trained models. The voice cloning extension of Bark, which is used by Bark-VC can be found [here](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/).

- ICASSP 2024 paper: https://ieeexplore.ieee.org/document/10447871
- GitHub repository: https://github.com/m-pana/spk_anon_nac_lm

## Installation

The only requirement for this pipeline is [Coqui TTS](https://docs.coqui.ai/en/latest/installation.html).

## Implementation

We have implemented the pipeline following the design principles of the framework. As outlined in the config file, it comprises the following components:

1. **Feature extractor** (`barkvc/hubert.py`): quantized base HuBERT trained on LibriSpeech.
2. **Feature processing** (`barkvc/converter.py`): comprises two language models and the EnCodec encoder, which convert the audio to the target speaker and represent it as EnCodec codes.
3. **Synthesis** (`barkvc/decoder.py`): EnCodec decoder.

## Training data

- HuBERT: LibriSpeech (960 h)
- EnCodec: clean speech from DNS Challenge 4 & Common Voice
- Bark: N/A