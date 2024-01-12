#!/bin/bash

mkdir -p ~/python_envs

python3 -m venv $HOME/python_envs/astro_waste_env
source $HOME/python_envs/astro_waste_env/bin/activate

pip3 install --upgrade pip
pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv-python gym
pip3 install keras_preprocessing --no-deps
pip3 install tensorflow
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install optax flax
pip3 install stable-baselines3 tensorboard
pip3 install --upgrade nvidia-cudnn-cu11 nvidia-cufft-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvcc-cu11 nvidia-cuda-runtime-cu11

deactivate

echo "alias activateWaste='source $HOME/python_envs/astro_waste_env/bin/activate'" >> ~/.bash_aliases