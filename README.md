# Speaker Anonymization Models

This collection of models are meant to be used with the evaluation framework [spkanon_eval](https://github.com/carlosfranzreb/spkanon_eval). The additional dependencies are defines in the corresponding build files in the `build` folder. The config and components of each model can be found under the model's folder.

Information and evaluation results for each model can be found in the `docs` folder.

## Current models

- **STT-TTS with Whisper & FastPitch**: does not require any further installation. Read more about this pipeline [here](docs/whisper_fastpitch.md).
- **StarGANv2-VC**: install it with `build/stargan.sh`. Read more about this pipeline [here](docs/whisper_fastpitch.md).
- **SoftVC**: install it with `build/softvc.sh` does not require any further installation. Read more about this pipeline [here](docs/whisper_fastpitch.md).