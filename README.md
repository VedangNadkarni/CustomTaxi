# CustomTaxi
An openai-gym environment based off of Taxi-v2. It has the capability to Dynamically generate a random map for reinforcement learning.

The original Taxi-v2 environment available from the openai-gym repo restricts the environment to limited dimensions. The environment is very promising since it is deterministic, uses text based graphics (reducing rendering overhead). However, it may be inadequate in providing a broader environment. Hence this custom environment has the following improvements over Taxi-v2 and Taxi-v3:
1. It has custom dimensions to give a larger sample space and exponentially increase the number of discrete states of the environment, which can help with more thorough training. *Beware however, that being a discrete environment, it creates an exhaustive sample space of all possible states which can increase exponentially when certain parameters are varied. This can lead to a slowdown in generation of the environment, and further in the convergence of the trained action model.* The environment can be modified in the following ways:
	1. Modify the dimensions of the 2D grid that the taxi can travel in.
	2. Modify the number of drop-off/pickup locations
	3. Modify the density of walls, that block the taxi from moving
	4. Modify the probability of all the drop-off locations being placed on the grid
2. It uses alphabets (characters starting from 0 \[0 character has an ASCII value of 55\]) instead of just RGBY

There is a lot more scope for improvement here, with more taxis, adding horizontal walls and a better probability of placing locations. It may also be possible to write a function to generate the environment in the code rather than modifying the environment file and then installing it again.

The dependencies added here are a lot more than actually needed to run this, since this was used in some other work with deep learning. The basic dependencies here are `numpy`, `setuptools` and `gym`. However, this hasn't been tested with a different, more minimal environment.

To install this follow the follownig tutorial http://www.gymlibrary.ml/pages/environment_creation/#example-custom-environment or just
1. Clone the repo
2. Install the dependecies
3. Use `pip install -e CustomTaxi-v2`

Then in the same environment, you may use the environment.
