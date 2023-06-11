from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.core.mission import MissionSpace
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType


import numpy as np

class BasicMazeEnv(MiniGridEnv):
    
    def _get_maze_matrix(self):
        return [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
        ]
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        maze_matrix = self._get_maze_matrix()
        self.agent_pos = self._agent_default_pos

        self.agent_dir = 0

        self.goal_pos = self._goal_default_pos
        self.put_obj(Goal(), *self.goal_pos)
        for i in range(len(maze_matrix)):
            for j in range(len(maze_matrix)):
                if maze_matrix[i][j]:
                    self.grid.set(j + 1, i + 1, Wall())
    

        # self.place_agent()

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def __init__(self, agent_pos=(1, 1), goal_pos=None, size = 12, old=False, flat=False, max_steps=80, agent_view_size=9, **kwargs):
        
        self.size = size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = np.array((self.size - 2, self.size - 2))

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.old = old
        self.flat = flat

        self.rewards = []
        self.path_lengths = []
        self.rotations = []
        self.collisions = []

        self.current_reward = 0
        self.current_path_length = 0
        self.current_rotations = 0
        self.current_collisions = 0


        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=True,
            highlight=False,
            **kwargs,
        )


    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        is_collision, is_rotation = False, True

        punish = 0.15
        rot = 3
        euclid_coeff, manhattan_coeff = 0.2, 5

        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
       

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
            reward -= punish/rot

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            reward -= punish/rot

        # Move forward
        elif action == self.actions.forward:
            is_rotation = False

            if fwd_cell is None or fwd_cell.can_overlap():
                fwd_pos = np.array(fwd_pos)
                euclid_change = np.linalg.norm(self._goal_default_pos - self.agent_pos, ord=2) - np.linalg.norm(self._goal_default_pos - fwd_pos, ord=2)
                manhattan_change = np.linalg.norm(self._goal_default_pos - self.agent_pos, ord=1) - np.linalg.norm(self._goal_default_pos - fwd_pos, ord=1)                  
                local_reward = euclid_coeff*euclid_change + manhattan_coeff*manhattan_change - 0.2
                reward += local_reward/self.max_steps
                self.agent_pos = tuple(fwd_pos)

            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward += self._reward()

            if fwd_cell is not None and fwd_cell.type == "wall":
                is_collision = True
                reward -= punish*2

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            reward -= 1
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        self.current_reward += reward
        self.current_path_length += 1
        self.current_collisions += 1 if is_collision else 0
        self.current_rotations += 1 if is_rotation else 0



        if self.old:
            return obs, reward, terminated or truncated, {}

        return obs, reward, terminated, truncated, {}
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 6 - 0.9 * (self.step_count / self.max_steps)

    
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        self.rewards.append(self.current_reward)
        self.path_lengths.append(self.current_path_length)
        self.rotations.append(self.current_rotations)
        self.collisions.append(self.current_collisions)

        if len(self.rewards) % 100 == 0:
            print('Episode',
                len(self.rewards),
                'Reward',
                np.mean(self.rewards[-10:]),)

        self.current_reward, self.current_path_length, self.current_rotations, self.current_collisions = 0, 0, 0, 0

        if self.old:
            return obs
           
        return obs, info
    
    
    def gen_obs(self, *args, **kwargs):
        obs = super().gen_obs(*args, **kwargs)
        if self.flat:
            return obs['image'].flatten()
        
        return obs['image']
    



class ComplexMazeEnv(BasicMazeEnv):
    def __init__(self, *args, **kwargs):
        return super().__init__(size=22, **kwargs)

    def _get_maze_matrix(self):
        return [
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ]


class SimpleMazeEnv(BasicMazeEnv):
    def __init__(self, *args, **kwargs):
        return super().__init__(size=7, **kwargs)

    def _get_maze_matrix(self):
        return [[0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0]]
