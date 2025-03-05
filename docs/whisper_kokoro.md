# Whisper + Kokoro

This pipeline is an STT-TTS, the same as `whisper_fastpitch`, but using Kokoro instead of FastPitch and HiFiGAN. Kokoro is one of the most popular TTS systems in HuggingFace, as well as one of the best in the TTS Arena. It has been trained on several voices, but they differ in quality. You can read more about this pipeline in HuggingFace or their GitHub page:

- HuggingFace: <https://huggingface.co/hexgrad/Kokoro-82M>
- GitHub: <https://github.com/hexgrad/kokoro>

To run this pipeline you only need to install kokoro: `pip install kokoro`.
