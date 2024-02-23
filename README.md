# TOXIC WASTE GYM

## Installation

In order to fully use the toxic waste gym library, you need to install a virtual python environment. To that end you can follow the next commands:

```
git clone https://github.com/anavc97/Toxic_Waste_Gym.git toxic_waste_env
cd toxic_was_env
./install_python_env.sh
```

After the script runs, the alias *activateWaste* is added to the bash aliases making it easier to activate the environment solely by running the alias.
If the script ends without any errors, you have the python environment fully installed. If you see any errors installing a specific package, try installing the specific packages isolated.



## Training the model

```
python3 train_astro_disposal_dqn.py --nagents 2 --nlayers 3 --buffer 50000 --cnn --tensorboard --layer-sizes 128 --iterations 200 --batch 128 --train-freq 10 --target-freq 1000 --field-size 15 15  --max-steps 200 --levels level_one --layer-obs --agent-centered --use-encoding --require_facing
```
