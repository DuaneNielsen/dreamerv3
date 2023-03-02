import gymnasium
from envs.gridworld import SimpleGridWorld, env_configs
import envs.simpler_gridworld

kwargs = {'grid': env_configs['grab_n_go']}

gymnasium.register(
    id="SimpleGridworld-grab_n_go-v0",
    entry_point="envs.gridworld:SimpleGridWorld",
    max_episode_steps=200,
    kwargs=kwargs
)

for world in envs.simpler_gridworld.worlds:
    gymnasium.register(
        id=f"SimplerGridWorld-{world}-v0",
        entry_point="envs.simpler_gridworld:SimplerGridWorld",
        max_episode_steps=100,
        kwargs={'world_name': world}
    )