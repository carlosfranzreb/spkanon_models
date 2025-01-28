conda activate ./venv
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy
pip install Cython
pip install "nemo_toolkit[tts]"
pip uninstall huggingface-hub
pip install huggingface-hub==0.23.2
python -m pip install torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install ./spkanon_eval
conda install -c conda-forge pynini
pip install nemo_text_processing