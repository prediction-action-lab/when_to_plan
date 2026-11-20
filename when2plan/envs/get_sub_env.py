from when2plan.envs.box_push.env import BoxPushEnv
from when2plan.envs.doorkey.env import DoorKeyEnv
from when2plan.envs.maze.env import MazeEnv
from when2plan.envs.fetch_pick_place.env import FetchPickPlaceEnv
from when2plan.envs.navigation.env import NavigationEnv


def get_sub_env_class(env_type):
    if env_type == "box_push":
        return BoxPushEnv
    elif env_type == "doorkey":
        return DoorKeyEnv
    elif env_type == "maze":
        return MazeEnv
    elif env_type == "fetch_pick_place":
        return FetchPickPlaceEnv
    elif env_type == "navigation":
        return NavigationEnv
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
