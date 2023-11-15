# kNN-VC

kNN-VC is a zero-shot voice conversion model based on WavLM. It computes the WavLM features of the source and target speech, and replaces the source features with the target features that are closest according to cosine similarity, resulting in arguably state-of-the-art zero-shot conversion.

- Interspeech 2023 paper: https://www.isca-speech.org/archive/interspeech_2023/baas23_interspeech.html
- GitHub repository: https://github.com/bshall/knn-vc

## Installation

To run this pipeline, you have to clone the download the WavLM and HiFiGAN checkpoints and add their paths to the config file.

- WavLM: https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt
- HifiGAN: https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt

## Implementation

We have implemented wrappers for its different components and included each in the appropriate phase. You can find the whole configuration for this model in the config file `knnvc/config.yaml`.

### Feature extraction

The feature extraction module comprises the WavLM model, implemented as a feature extractor in the evaluation framework (`spkanon_eval/featex/wavlm`). Its config must include the following fields:

- `ckpt`: path to the checkpoint.
- `layer`: layer from which the representation is returned. In the paper, they state layer 6 works best.
- `hop_length`: used to compute the number of features extracted per input sample. It should be left as 320.

### Feature processing

This module comprises the nearest-neighbors conversion. For that, it must first compute the WavLM features of the target speakers. It does so in the constructor, with the datasets defined in the dataset config under `targets`. To compute the target features, the following fields are required in the config:

- `n_neighbors`: the number of neighbors that are averaged to compute the target feature for a source feature. In the paper they use 4 neighbors.
- `wavlm`: configuration of the WavLM feature extractor, as defined above.
- `wavlm_dl`: batch size, num. of workers and sample rate to use for the dataloader.
- `exp_folder`: current experiment folder, used to find the target datafile and to dump the computed WavLM features. These features can be re-used by a future run by setting their path in the field `wavlm_feats`.


### Synthesis

A HifiGAN model trained on converted WavLM features is used as a vocoder. Its configuration requires the following fields:

- `ckpt`: path to the model checkpoint.
- `config`: path to the config file, which can be also be downloaded from the GitHub repository (<https://github.com/bshall/knn-vc/blob/master/hifigan/config_v1_wavlm.json>).
- `hop_length`: upsampling ratio of the HiFiGAN. It is used to compute the durations of the synthesized audio. In the default setting, this value is the same as the hop length of the WavLM model (320).
