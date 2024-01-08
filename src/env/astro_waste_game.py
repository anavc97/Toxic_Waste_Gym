#! /usr/bin/env python

import sys
import numpy as np
import gym
import pickle
import yaml
import itertools
import random
import time
import os
import copy

from pathlib import Path
from termcolor import colored
from typing import List, Tuple, Callable, Dict
from time import time
from abc import ABC, abstractmethod
from threading import Lock, Thread
from queue import Queue, LifoQueue, Empty, Full


class AstroWasteGame(ABC):
	_players: Dict[str, int]
	_player_actions: Dict[str, int]
	_game_id: int
	_is_active: bool
	_cycles_second: int
	_maximum_players: int
	