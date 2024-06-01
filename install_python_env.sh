#!/bin/bash

CUDA_VERSION=$(nvidia-smi | grep 'CUDA Version' | awk -F" " '{print $9}' | awk -F"." '{print $1}')

if [ "$1" = "conda" ]; then

  if ! command -v mamba &> /dev/null; then
    conda install -y conda-forge::mamba
    mamba init
    source "$HOME"/.bashrc
  else
    if [ -z "$CONDA_SHLVL" ]; then
        mamba init
        source "$HOME"/.bashrc
    fi
  fi

  mamba create -y -n toxic_waste_env python=3.10
  conda activate toxic_waste_env

  mamba install -y -c conda-forge numpy scipy matplotlib pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv
  mamba install -y -c conda-forge stable-baselines3 tensorboard wandb gym pyglet
  pip3 install --upgrade "jax[cuda"$CUDA_VERSION"_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  mamba install -y -c conda-forge optax flax
  pip3 install --upgrade nvidia-cudnn-cu11 nvidia-cufft-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvcc-cu11 nvidia-cuda-runtime-cu11

  conda deactivate

  {
    echo "alias activateWaste=\"conda activate toxic_waste_env\""
    echo "alias astro_toxic=\"conda activate toxic_waste_env; cd project/toxic_waste\""
  } >> ~/.bash_aliases

  env_home="$HOME"/miniconda3/envs/toxic_waste_env
  touch "$env_home"/etc/conda/activate.d/env_vars.sh
  touch "$env_home"/etc/conda/deactivate.d/env_vars.sh
  {
    echo "#!/bin/sh"
    echo "EXTRA_PATH=\"\$HOME/project/toxic_waste/src:\""
    echo "OLD_PYTHONPATH=\"$PYTHONPATH\""
    echo "PYTHONPATH=$EXTRA_PATH:$PYTHONPATH"
    echo "export PYTHONPATH"
    echo "export OLD_PYTHONPATH"
  } >> "$env_home"/etc/conda/activate.d/env_vars.sh

  {
    echo "#!/bin/sh"
    echo "PYTHONPATH=\"$OLD_PYTHONPATH\""
    echo "unset OLD_PYTHONPATH"
  } >> "$env_home"/etc/conda/deactivate.d/env_vars.sh

else

  mkdir -p ~/python_envs
  python3.10 -m venv "$HOME"/python_envs/toxic_waste_env
  source "$HOME"/python_envs/toxic_waste_env/bin/activate

  pip3 install --upgrade pip
  pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose pyyaml termcolor tqdm scikit-learn opencv-python gym pyglet
  pip3 install --upgrade "jax[cuda"$CUDA_VERSION"_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  pip3 install optax flax
  pip3 install stable-baselines3 tensorboard
  pip3 install --upgrade nvidia-cudnn-cu11 nvidia-cufft-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvcc-cu11 nvidia-cuda-runtime-cu11

  deactivate

  {
    echo "alias activateWaste=\"source \$HOME/python_envs/toxic_waste_env/bin/activate\""
    echo "alias astro_toxic=\"source \$HOME/python_envs/toxic_waste_env/bin/activate; cd project/toxic_waste\""
  } >> ~/.bash_aliases

  {
    echo "EXTRA_PATH=\"\$HOME/project/toxic_waste/src:\""
    echo "PYTHONPATH=$EXTRA_PATH:$PYTHONPATH"
    echo "export PYTHONPATH"
  } >> "$HOME"/python_envs/toxic_waste_env/bin/activate

fi

source "$HOME"/.bashrc