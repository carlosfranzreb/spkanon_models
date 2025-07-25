# Speaker Anonymizers

This collection of models are meant to be used with the evaluation framework [spkanon_eval](https://github.com/carlosfranzreb/spkanon_eval).
The additional dependencies are defined in the corresponding build files in the `build` folder.
The config and components of each anonymizer can be found under the anonymizer's folder.

Information and evaluation results for each anonymizer can be found below.

## Anonymizers

- [STT-TTS with Whisper & FastPitch](#stt-tts-with-whisper--fastpitch)
- [STT-TTS with Whisper & Kokoro](#stt-tts-with-whisper--kokoro)
- [StarGANv2-VC](#starganv2-vc)
- [Soft-VC](#soft-vc)
- [kNN-VC](#knn-vc)
- [Bark-VC](#bark-vc)
- [ASR-BN](#asr-bn)
- [Private kNN-VC](#private-knn-vc)

## STT-TTS with Whisper & FastPitch

This pipeline is an STT-TTS pipeline: input speech is transcribed (speech-to-text) and afterwards synthesized (text-to-speech).
The voice identity of the input speaker is removed through the transcription, resulting in perfect anonymization, but prosody and emotion are also removed.
This pipeline is the simplest implementation of this approach, using existing models without any fine-tuning.
The ASR model Whisper is used to transcribe the speech, which is then synthesized with NeMo's multi-speaker TTS pipeline.

### Installation

According to the NVIDIA's [NeMo](https://github.com/NVIDIA/NeMo) documentation, NeMo needs to be installed with `pip install nemo_toolkit`, and also the specific packages for the TTS models with `pip install nemo_toolkit['tts']`.
The models will be downloaded on runtime when the pipeline is used for the first time.
It requires Python 3.10 or above!

I've installed in my mac and in a linux machine in conda environments.
You can find the commands I ran in `build/whisper_fastpitch_macos.sh` and `build/whisper_fastpitch_debian.sh`, respectively.
In general, it is easier to install NeMo first and then the framework.
NeMo is always a pain to install, as its lists of dependencies is endless.
If you find a better way of getting these models to run, please let me know!

### Implementation

We have implemented wrappers for its different components and included each in the appropriate phase.
You can find the whole configuration for this model in `whisper_fastpitch/config.yaml`.

The **feature extraction module** comprises the ASR model Whisper, which predicts the text contained in the input speech.
Our wrapper for this model is described [in this doc file](components/featex/asr.md), and implemented in `src/featex/asr/whisper.py`.
Here are the links to the pre-print and the GitHub repository:

- [Whisper paper](https://arxiv.org/pdf/2212.04356)
- [Whisper repository](https://github.com/openai/whisper)

NeMo's FastPitch is the sole component of the **feature processing module**.
It receives text as input and outputs a spectrogram, which corresponds to the text uttered by one of the 20 target speakers that are available.
The target speakers result from interpolating 10 speakers from HiFiTTS, which the model was trained on.
The HiFiGAN synthesizer was trained alongside FastPitch, and described in the same model card.

- [FastPitch+ HiFiGAN model card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/tts_en_multispeaker_fastpitchhifigan)

### Training data

Whisper is trained on 680k hours of multilingual data scraped from the Internet.
The exact dataset is unknown.
FastPitch and HiFiGAN were trained together on 291 hours of 10 speakers from HiFiTTS.

## STT-TTS with Whisper & Kokoro

: requires installing [Kokoro](https://github.com/hexgrad/kokoro)

As the one above, this pipeline is an STT-TTS, but using Kokoro instead of FastPitch and HiFiGAN.
Kokoro is one of the [most popular TTS systems in HuggingFace](https://huggingface.co/models?pipeline_tag=text-to-speech), as well as one of the best in the [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2).
It has been trained on several voices, but they differ in quality.
You can read more about this pipeline in HuggingFace or their GitHub page:

- [Kokoro HuggingFace page](https://huggingface.co/hexgrad/Kokoro-82M)
- [Kokoro repository](https://github.com/hexgrad/kokoro)

### Installation

To run this pipeline, you only need to install kokoro: `pip install kokoro`.

### Training data

Whisper is trained on 680k hours of multilingual data scraped from the Internet.
The exact dataset is unknown.
The training data of Kokoro is also [undisclosed](https://huggingface.co/hexgrad/Kokoro-82M#training-details).

## StarGANv2-VC

StarGANv2-VC is an unsupervised voice converter, where the input speech is adapted to conform with a given style vector.
The style vector is extracted from the mapping network with the target ID.
The generator receives the style vector of the target, as well as the spectrogram and the F0 contour of the input speech.
It converts the input spectrogram to the given style vector, preserving the F0 contour, which is then synthesized with the Parallel WaveGAN.
The generator is trained adversarially with two models: one which discriminates real samples from fake ones, and another which recognizes the converted sample's speaker.
Six additional losses are included to ensure that the style vector is used appropriately, and that the converted speech is consistent with the input speech.

- [StarGANv2-VC paper](https://www.isca-speech.org/archive/interspeech_2021/li21e_interspeech.html)
- [StarGANv2-VC repository](https://github.com/yl4579/StarGANv2-VC)

### Installation

To run this pipeline, you have to clone the original StarGANv2-VC repository and download the weights.
You can do so by running the build file `build/stargan.sh`.

There is an import error that must be fixed manually, as explained in [this issue](https://github.com/kan-bayashi/ParallelWaveGAN/issues/430).
You must change the line `from scipy.signal import kaiser` to `from scipy.signal.windows import kaiser` in the file `venv/lib/python3.11/site-packages/parallel_wavegan/layers/pqmf.py`.

### Implementation

We have implemented wrappers for its different components and included each in the appropriate phase.
You can find the whole configuration for this model in `stargan/config.yaml`.

The feature extraction module comprises the spectrogram extractor.

The feature processing module comprises the StarGANv2-VC conversion component.
It receives spectrograms and the source speaker labels as input, and optionally the desired target speaker.
It outputs the converted spectrogram, conditioned on the target speaker.
 We use the mapping network to extract the style vectors of the target speakers.

The Parallel WaveGAN synthesizer comes from a [Python package](https://pypi.org/project/parallel-wavegan/).
We implement a wrapper for it, which receives the converted spectrogram as input and outputs the synthesized waveform.

### Training data

The model was trained on 20 speakers of VCTK, a corpus of read speech recorded in a professional studio.
The exact dataset can be found in the GitHub repository, linked above.

## Soft-VC

Soft-VC is a voice conversion model based on HuBERT.
It computes the HuBERT features of the input speech, which implicitly perform the anonymization, given that the HuBERT model was trained to predict text, and not the speaker.
They improve the preservation of the linguistic content by predicting the distribution over the discrete HuBERT units, resulting in what they term soft speech units.
These units capture more paralinguistics and improve the intelligibility and naturalness of the converted speech.
In the Soft-VC pipeline, HuBERT is followed by an acoustic model that transforms HuBERT features into a spectrogram, which is then synthesized by HiFI-GAN.
The HiFi-GAN they used is only trained on one speaker from LibriTTS, making Soft-VC an any-to-one voice conversion model.

- [Soft-VC paper](https://ieeexplore.ieee.org/abstract/document/9746484)
- [Soft-VC repository](https://github.com/bshall/soft-vc)

### Installation

Nothing needs to be installed manually.
The models and weights will be downloaded automatically on runtime through `torch.hub`.

### Implementation

We have implemented wrappers for its different components and included each in the appropriate phase.
You can find the whole configuration for this model in `softvc/config.yaml`.

The feature extraction module comprises the HuBERT model, followed by a linear projection to predict probabilities over the different speech units.
This component is implemented in a separate [GitHub repository](https://github.com/bshall/hubert).
Our wrapper can be found in the file `softvc/hubert_softvc.py`.

This module comprises the acoustic model that transforms the HuBERT units to a spectrogram.
It is implemented in a separate [GitHub repository](https://github.com/bshall/acoustic-model).
It is trained on only one speaker, and therefore provides only one target voice.
Our wrapper can be found in the file `softvc/acoustic_softvc.py`.

The feature processing module comprises the acoustic model that transforms the HuBERT units to a spectrogram.
It is implemented in a separate [GitHub repository](https://github.com/bshall/hifigan), and trained with the output of the acoustic model.
Our wrapper can be found in the file `softvc/hifigan_softvc.py`.

### Training data

HuBERT is a self-supervised model trained on the whole [Librispeech](https://www.openslr.org/12) dataset (960 h).
The acoustic model and the vocoder are trained with one speaker of [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (24 h).
The exact train and test splits can be found [here](https://github.com/bshall/hifigan/releases/tag/v0.1).
The acoustic model is trained for 50k steps.
The checkpoint with the lowest validation loss is chosen.
The HiFiGAN is trained with ground-truth spectrograms for 1M steps and then fine-tune on predicted spectrograms for 500k steps.

## kNN-VC

kNN-VC is a zero-shot voice conversion model based on WavLM.
It computes the WavLM features of the source and target speech, and replaces the source features with the target features that are closest according to cosine similarity, resulting in arguably state-of-the-art zero-shot conversion.

- [kNN-VC paper](https://www.isca-speech.org/archive/interspeech_2023/baas23_interspeech.html)
- [GitHub repository](https://github.com/bshall/knn-vc)

### Installation

To run this pipeline, you have to download the WavLM and HiFiGAN checkpoints and add their paths to the config file.

- [WavLM checkpoint](https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt)
- [kNN-VC's HifiGAN checkpoint](https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt)

### Implementation

We have implemented wrappers for its different components and included each in the appropriate phase.
You can find the whole configuration for this model in the config file `knnvc/config.yaml`.

The feature extraction module comprises [WavLM](https://arxiv.org/pdf/2110.13900), implemented as a feature extractor in the evaluation framework (`spkanon_eval/featex/wavlm`).
Its config must include the following fields:

- `ckpt`: path to the checkpoint.
- `layer`: layer from which the representation is returned. In the paper, they state layer 6 works best.
- `hop_length`: used to compute the number of features extracted per input sample. It should be left as 320.

This module comprises the nearest-neighbors conversion.
For that, it must first compute the WavLM features of the target speakers.
It does so in the constructor, with the datasets defined in the dataset config under `targets`.
To compute the target features, the following fields are required in the config:

- `n_neighbors`: the number of neighbors that are averaged to compute the target feature for a source feature.
In the paper they use 4 neighbors.
- `wavlm`: configuration of the WavLM feature extractor, as defined above.
- `wavlm_dl`: batch size, num. of workers and sample rate to use for the dataloader.
- `exp_folder`: current experiment folder, used to find the target datafile and to dump the computed WavLM features.
These features can be re-used by a future run by setting their path in the field `wavlm_feats`.

A HifiGAN model trained on converted WavLM features is used as a vocoder.
Its configuration requires the following fields:

- `ckpt`: path to the model checkpoint.
- `config`: path to the config file, which can be also be downloaded from the GitHub repository (<https://github.com/bshall/knn-vc/blob/master/hifigan/config_v1_wavlm.json>).
- `hop_length`: upsampling ratio of the HiFiGAN. It is used to compute the durations of the synthesized audio. In the default setting, this value is the same as the hop length of the WavLM model (320).

### Training data

[Wavlm](https://arxiv.org/pdf/2110.13900) was trained with LibriLight, GigaSpeech and VoxPopuli (94k hours in total).
The HiFiGAN synthesizer was trained by the authors on [Librispeech](https://www.openslr.org/12).

## Bark-VC

Bark-VC adapts the VALL-E approach, which is a text-to-speech model, to perform voice conversion.
Suno AI open sourced an implementation of VALL-E, called [Bark](https://github.com/suno-ai/bark/blob/main/model-card.md).
It differs from VALL-E in that it represents the input text with [HuBERT](https://arxiv.org/pdf/2106.07447) features instead of phonemes, enabling its adaptation for voice conversion: instead of computing the HuBERT features from text, they are extracted from the input audio.
The rest of the pipeline stays intact, allowing this approach to use the pre-trained models.
The voice cloning extension of Bark, which is used by Bark-VC can be found [here](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/).

- [Bark-VC paper](https://ieeexplore.ieee.org/document/10447871) (there is also an open-access pre-print on arxiv)
- [GitHub repository](https://github.com/m-pana/spk_anon_nac_lm)

### Installation

The only requirement for this pipeline is [Coqui TTS](https://docs.coqui.ai/en/latest/installation.html).

### Implementation

We have implemented the pipeline following the design principles of the framework. As outlined in the config file, it comprises the following components:

1. **Feature extractor** (`barkvc/hubert.py`): quantized base HuBERT trained on [Librispeech](https://www.openslr.org/12).
2. **Feature processing** (`barkvc/converter.py`): comprises two language models and the EnCodec encoder, which convert the audio to the target speaker and represent it as EnCodec codes.
3. **Synthesis** (`barkvc/decoder.py`): EnCodec decoder.

### Training data

[HuBERT](https://arxiv.org/pdf/2106.07447) is trained with [Librispeech](https://www.openslr.org/12) (960 h)
[EnCodec] is trained with clean speech from DNS Challenge 4 & Common Voice
I couldn't find the training data of Bark.

## ASR-BN

ASR-BN is baseline B5 from the VoicePrivacy Challenge 2024.

- [VPC24 evaluation plan](https://www.voiceprivacychallenge.org/vp2024/docs/VoicePrivacy_2024_Eval_Plan_v2.1.pdf#page=12.26)
- [GitHub repository](https://github.com/deep-privacy/SA-toolkit)

### Installation

This anonymizer is downloaded automatically with `torch.hub`.
It does not require any additional packages.

### Implementation

Our implementation uses the whole anonymizer at once through torch.hub (see `asrbn/model.py`).
The target selection is done in `asrbn/featproc.py`.
There are 247 available targets from LibriTTS's train-clean-100 subset.

### Training data

[wav2vec2](https://arxiv.org/pdf/2006.11477) was trained on VoxPopuli (24k hours) and then fine-tuned on [Librispeech](https://www.openslr.org/12) train-clean-100.
The HiFiGAN model is trained on LibriTTS-train-clean-100 for each target speaker.

## Private kNN-VC

Private kNN-VC is an interpretable extension of kNN-VC, improving its privacy by restricting phonetic variation and altering phone durations.

- [Private kNN-VC paper](https://arxiv.org/pdf/2505.17584)
- [Private kNN-VC samples](https://carlosfranzreb.github.io/private-knnvc)
- [Private kNN-VC repository](https://github.com/carlosfranzreb/private_knnvc)

### Installation

To run this anonymizer, you have to download the kNN-VC requirements:

- [WavLM checkpoint](https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt)
- [kNN-VC's HifiGAN checkpoint](https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt)

Then you also need the checkpoints of the phone predictor and the duration predictor of private kNN-VC:

- [Private kNN-VC checkpoints](https://github.com/carlosfranzreb/private_knnvc/releases/download/v1.0.0/checkpoints.zip)

Set the paths to each checkpoint in the configuration file.

### Implementation

The implementation is similar to that of KNN-VC.
The only difference is the two new predictors, which are used by the converter to alter phone durations and restrict their variation.

### Training data

- [Wavlm](https://arxiv.org/pdf/2110.13900) was trained with LibriLight, GigaSpeech and VoxPopuli (94k hours in total).
- kNN-VC's HiFiGAN synthesizer was trained by the authors on [Librispeech](https://www.openslr.org/12) (960 h).
- Private kNN-VC's phone predictor was trained on [Librispeech](https://www.openslr.org/12) train-clean-100 (100 h).
- Private kNN-VC's duration predictor with a single speaker of [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (24 h).
