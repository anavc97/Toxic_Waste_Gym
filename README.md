# TOXIC WASTE GYM

The Toxic Waste environment is a collaborative environment where a robot agent has to cooperate with a human agent to collect toxic waste spread around the environment.
In this environment, both agents have to fully cooperate in order to clean the waste, this because the robot is the only agent that can safely hold the toxic waste, however it
doesn't have any arms to collect the trash items. So the human has to pick the trash items and place them in the robot's storage.

## Installation

In order to fully use the toxic waste gym library, you need to install a virtual python environment. To that end you can follow the next commands:

```
git clone https://github.com/anavc97/Toxic_Waste_Gym.git toxic_waste_env
cd toxic_was_env
```

to download and enter the directory with the environment's code. Then you will have to execute the *instal_python_env.sh* script to create the virtual environment.
The script supports both installation using *pip* or *anaconda*. To use *pip* execute the script as follows:

```
./install_python_env.sh
```

To use *anaconda* execute the script as such:
```
source ./install_python_env.sh conda
```

After the script runs, the alias *activateWaste* is added to the bash aliases making it easier to activate the environment solely by running the alias.
If the script ends without any errors, you have the python environment fully installed. If you see any errors installing a specific package, try installing the specific packages isolated.


## Training and testing models

In the *scripts* directory you can find scripts to both train and testing models. 

- ``` run_train_astro_disposal.py ``` - trains a model for the robot agent, using a predifined model for the human behaviour
- ``` run_train_toxic_central_model.py ``` - trains two seperate models, one for the human agent and another for the robot agent.
- ``` run_train_toxic_multi_model.py ``` - trains one joint model for the behaviour of both the human and robot agent
