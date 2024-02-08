#! /usr/bin/env python

import numpy as np
import pyglet
import six
import json

from gym import error
from .toxic_waste_env_v1 import AstroWasteEnv, CellEntity, HoldState, AgentType, ActionDirection
from typing import Tuple
from pathlib import Path
from pyglet.gl import *


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error("Invalid display specification: {}. (Must be a string like :0 or None.)".format(spec))


class Viewer(object):
    
    def __init__(self, world_size: Tuple[int, int], grid_size: int=32, icon_size: int=32, visible: bool=True):
        display = get_display(None)
        self.rows, self.cols = world_size
        self.grid_size = grid_size
        self.icon_size = icon_size
        self.w_grid_size = grid_size
        self.h_grid_size = grid_size

        self.width = self.cols * grid_size + 1
        self.height = self.rows * grid_size + 1
        self.window = pyglet.window.Window(width=self.width, height=self.height, display=display, visible=visible, resizable=True)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        icons_dir = Path(__file__).parent.absolute() / 'data' / 'icons'

        pyglet.resource.path = [str(icons_dir)]
        pyglet.resource.reindex()
        
        self.astro_imgs = dict([(x.stem, pyglet.resource.image(x.name)) for x in Path.iterdir(icons_dir) if x.stem.find('astro') != -1])
        self.human_imgs = dict([(x.stem, pyglet.resource.image(x.name)) for x in Path.iterdir(icons_dir) if x.stem.find('human') != -1])
        self.ball_img = pyglet.resource.image('ball.png')
        self.counter_img = pyglet.resource.image('counter.png')
        self.floor_img = pyglet.resource.image('floor.png')
        self.ice_img = pyglet.resource.image('ice.png')
        self.toxic_img = pyglet.resource.image('toxic.png')
        self.gaips_img = pyglet.resource.image('logo.png')
        self.project_img = pyglet.resource.image('logo_2.png')
        
        
    def close(self):
        self.window.close()
    
    def window_closed_by_user(self):
        self.isopen = False
        exit()
    
    def render(self, env: AstroWasteEnv, return_rgb_array: bool=False):
        glClearColor(0, 0, 0, 0)
        self.window.switch_to()
        self.window.clear()
        self.window.dispatch_events()
        
        if self.window.width != self.width or self.window.height != self.height:
            self.width = self.window.width
            self.height = self.window.height
            self.w_grid_size = self.window.width / self.cols
            self.h_grid_size = self.window.height / self.rows
        
        self._draw_world(env)
        self._draw_balls(env)
        self._draw_players(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen
    
    def _draw_world(self, env: AstroWasteEnv):
        batch = pyglet.graphics.Batch()
        terrains = []
        for col in range(self.cols):
            for row in range(self.rows):
                if env.field[row, col] == CellEntity.COUNTER:
                    img = self.counter_img
                elif env.field[row, col] == CellEntity.ICE:
                    img = self.ice_img
                elif env.field[row, col] == CellEntity.TOXIC:
                    img = self.toxic_img
                else:
                    img = self.floor_img
                terrains.append(pyglet.sprite.Sprite(img, self.w_grid_size * col, self.window.height - self.h_grid_size * (row + 1), batch=batch))
        for t in terrains:
            t.update(scale=self.w_grid_size / t.width)
        batch.draw()

    def _draw_balls(self, env: AstroWasteEnv):
        balls = []
        batch = pyglet.graphics.Batch()

        for ball in env.objects:
            if ball.hold_state == HoldState.FREE:
                row, col = ball.position
                balls.append(pyglet.sprite.Sprite(self.ball_img, self.w_grid_size * col, self.window.height - self.h_grid_size * (row + 1), batch=batch))
        for ball in balls:
            ball.update(scale=self.w_grid_size / ball.width)
        batch.draw()

    def _draw_players(self, env: AstroWasteEnv):
        players = []
        batch = pyglet.graphics.Batch()

        for player in env.players:
            row, col = player.position
            orientation = player.orientation
            if player.agent_type == AgentType.HUMAN:
                sprite_name = 'human-%s%s' % (ActionDirection(orientation).name, '-ball' if player.is_holding_object() else '')
                sprite = self.human_imgs[sprite_name]
            else:
                sprite_name = 'astro-%s' % ActionDirection(orientation).name
                sprite = self.astro_imgs[sprite_name]
            players.append(pyglet.sprite.Sprite(sprite, self.w_grid_size * col, self.window.height - self.h_grid_size * (row + 1), batch=batch))
        for p in players:
            p.update(scale=self.w_grid_size / p.width)
        batch.draw()