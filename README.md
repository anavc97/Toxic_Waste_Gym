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
3. Open terminal and run ``` python3.10 socket_game_server.py --field-size 15 15 --max-env-players 2 --max-game-players 2 --max-objects 6 --max-steps 200 --cycles-second 5 --game-id 000 --levels level_one --layer-obs --centered-obs ```
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

