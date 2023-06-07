from gymnasium import register
from . import mazes


register(
    id='SimpleMazeEnv-v0',
    entry_point="environments.mazes:SimpleMazeEnv",
    kwargs={},
)

register(
    id='BasicMazeEnv-v0',
    entry_point="environments.mazes:BasicMazeEnv",
    kwargs={},
)

register(
    id='ComplexMazeEnv-v0',
    entry_point="environments.mazes:ComplexMazeEnv",
    kwargs={},
)