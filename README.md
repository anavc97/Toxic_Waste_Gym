<<<<<<< HEAD
# TOXIC WASTE GYM

The Toxic Waste environment is a collaborative environment where a robot agent has to cooperate with a human agent to collect toxic waste spread around the environment.
In this environment, both agents have to fully cooperate in order to clean the waste, this because the robot is the only agent that can safely hold the toxic waste, however it
doesn't have any arms to collect the trash items. So the human has to pick the trash items and place them in the robot's storage.

## Installation

In order to fully use the toxic waste gym library, you need to install a virtual python environment. To that end you can follow the next commands:

```
git clone https://github.com/anavc97/Toxic_Waste_Gym.git toxic_waste
cd toxic_waste
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
=======
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


## Running game and server

1. Start virtual environment (*activateWaste*)
2. Open Unity in Scene "Intro"
3. Open terminal and run ``` python3.10 socket_game_server.py --field-size 15 15 --max-env-players 2 --max-game-players 2 --max-objects 6 --max-steps 200 --cycles-second 5 --game-id 000 --levels level_one level_two --layer-obs --centered-obs --nlayers 2 --buffer 1000 --layer-size 256 --version 2 --cnn --vdn --ddqn --tau 0.1 ```
4. Run Unity Scene "Intro"
5. Press Start button

Server and Unity should be synced - server should be sending states, Unity should be sending actions

# To test local web implementation:

1. Build game with WebGL (In player Settings >> Publishing Settings >> Compression format, chosse Disabled)
2. Go to the Build folder
3. Run ```python3 -m http.server```
4. Open localhost:8000 on browser

## Common errors: 

1. *Socket Error: Address already in use*
**solution:**  usually works to just wait a bit.

## Current issues

>>>>>>> origin
