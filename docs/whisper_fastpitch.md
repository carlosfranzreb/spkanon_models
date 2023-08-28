# Whisper + FastPitch + HifiGAN

This pipeline is an STT-TTS pipeline: input speech is transcribed (speech-to-text) and afterwards synthesized (text-to-speech). The voice identity of the input speaker is removed through the transcription, but the prosody and paralinguistic information contained in the input speech is also removed. This pipeline is the simplest implementation of this approach, using existing models without any fine-tuning. The ASR model Whisper is used to transcribe the speech, which is then synthesized with NeMo's multi-speaker TTS pipeline.

## Installation

Nothing needs to be installed, as NeMo is already part of the framework. The models will be downloaded on runtime when the pipeline is used for the first time.

## Implementation

We have implemented wrappers for its different components and included each in the appropriate phase. You can find the whole configuration for this model in `config/pipelines/whisper_fastpitch.yaml`. The results shown below use the small Whisper model, but larger models can be selected through the configuration.

### Feature extraction

The feature extraction module comprises the ASR model Whisper, which predicts the text contained in the input speech. Whisper is an encoder-decoder Transformer trained on 680k hours of multilingual speech scraped from the Internet. Our wrapper for this model is described [here](components/featex/asr.md), and written in the file `src/featex/asr/whisper.py`.

### Feature processing

NeMo's FastPitch is the sole component of the processing module. It receives text as input and outputs a spectrogram. You can find the model card [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/tts_en_multispeaker_fastpitchhifigan). It comprises 20 target speakers which result from interpolating 10 speakers from HiFiTTS, which the model was trained on. If a target speaker is not given, it uses its own target selection algorithm to pick a target. You can read more about target selection algorithms [here](components/target_selection.md).

### Synthesis

The HiFiGAN synthesizer was trained alongside FastPitch, and described in the same model card.

## Training data

Whisper is trained on 680k hours of multilingual data scraped from the Internet. The exact dataset is unknown. FastPitch and HiFiGAN were trained together on 291 hours of 10 speakers from HiFiTTS.

## Results

We have evaluated these models with the setup described in the SPSC 2023 paper.

### Evaluation results

| | cv-test_3utts | ls-test-clean | edacc-test | ravdess |
| --- | --- | --- | --- | --- |
| whisper-small | 0.19 | 0.08 | 0.34 | 0.01 |
| whisper-large | 0.18 | 0.08 | 0.34 | 0.01 |
| ser-audeering-w2v | 1.0 | 0.99 | 0.99 | 1.0 |
| nisqa | 3.87 | 3.7 | 3.58 | 3.62 |
| asv-plda/ignorant/results | 0.5 | 0.48 | 0.49 | 0.48 |
| asv-plda/lazy-informed/results | 0.47 | 0.43 | 0.47 | 0.51 |

### Inference time for 10s input duration (in seconds)

| Component | inference time |
| --- | --- |
| cpu_inference | 1.74 |
| cuda_inference | 0.09 |
