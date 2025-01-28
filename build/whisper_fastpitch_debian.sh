export PATH="/netscratch/franzreb/miniconda3/bin:$PATH"
conda create -p ./venv_nemo python=3.10.12
conda activate ./venv_nemo
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install nemo_toolkit['all']
pip install ./spkanon_eval/