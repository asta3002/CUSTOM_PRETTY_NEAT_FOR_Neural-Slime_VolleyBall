import numpy as np
import gym
import slimevolleygym
from matplotlib.pyplot import imread
# from SlimeVolleyEnv import SlimeEnv
class SurvivalRewardEnv(gym.RewardWrapper):
  """
  A RewardWrapper for Gymnasium environments that adds a small
  survival bonus to the reward at each timestep.

  This encourages the agent to prolong the episode.
  """
  def __init__(self, env):
    """
    Initializes the SurvivalRewardEnv wrapper.

    :param env: (Gymnasium Environment) The environment to wrap.
    """
    super().__init__(env) # Preferred way to call parent constructor in Python 3+
    print("SurvivalRewardEnv initialized: Adding +0.01 reward per timestep.")

  def reward(self, reward):
    """
    Modifies the reward by adding a survival bonus.

    :param reward: (float) The original reward from the wrapped environment.
    :return: (float) The modified reward.
    """
    # Add a small positive constant to the reward for each timestep
    return reward + 0.0003

def make_env(env_name, seed=-1, render_mode=False,testing_mode=False):
  # -- Bullet Environments ------------------------------------------- -- #
  if "Bullet" in env_name:
    import pybullet as p # pip install pybullet
    import pybullet_envs
    import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv

  # -- Bipedal Walker ------------------------------------------------ -- #
  if (env_name.startswith("BipedalWalker")):
    if (env_name.startswith("BipedalWalkerHardcore")):
      import Box2D
      from domain.bipedal_walker import BipedalWalkerHardcore
      env = BipedalWalkerHardcore()
    elif (env_name.startswith("BipedalWalkerMedium")): 
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()
      env.accel = 3
    else:
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()


  # -- VAE Racing ---------------------------------------------------- -- #
  elif (env_name.startswith("VAERacing")):
    from domain.vae_racing import VAERacing
    env = VAERacing()
    
    
  # -- Classification ------------------------------------------------ -- #
  elif (env_name.startswith("Classify")):
    from domain.classify_gym import ClassifyEnv
    if env_name.endswith("digits"):
      from domain.classify_gym import digit_raw
      trainSet, target  = digit_raw()
        
    if env_name.endswith("mnist256"):
      from domain.classify_gym import mnist_256
      trainSet, target  = mnist_256()

    env = ClassifyEnv(trainSet,target)  


  # -- Cart Pole Swing up -------------------------------------------- -- #
  elif (env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
    if (env_name.startswith("CartPoleSwingUp_Hard")):
      env.dt = 0.01
      env.t_limit = 200
  elif (env_name.startswith("SlimeVolley")):
    
    og_env = gym.make(env_name)
    # og_env = SlimeEnv(render_mode="human")
    if not testing_mode:
      env = SurvivalRewardEnv(og_env)
      print("Up&Running")
    else:
      
      env = og_env
    
  # -- Other  ------------------------------------------------------- -- #
  else:
    env = gym.make(env_name)

  if (seed >= 0):
    domain.seed(seed)

  return env