from .builder import ENVIRONMENTS
import gymnasium as gym

@ENVIRONMENTS.register_module()
class Environments(gym.Env):
    def __init__(self, **kwargs):
        super(Environments, self).__init__()