from gym.envs.registration import register

register(id='CustomTaxi2-v2', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='custom_envs.envs:CustomTaxiEnv' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)