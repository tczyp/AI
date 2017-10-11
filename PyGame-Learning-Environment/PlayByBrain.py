import numpy as np
# from ple.games.pong import Pong as MyGame
# from ple.games.pixelcopter import Pixelcopter as MyGame
# from ple.games.puckworld import PuckWorld as MyGame
# from ple.games.raycastmaze import RaycastMaze as MyGame
# from ple.games.snake import Snake as MyGame
# from ple.games.waterworld import WaterWorld as MyGame
from ple.games.flappybird import FlappyBird as MyGame
# from ple.games.monsterkong import MonsterKong as MyGame
from ple import PLE
import cv2
from BrainDQN import BrainDQN

class NaiveAgent():
	"""
		This is our naive agent. It picks actions at random!
	"""
	def __init__(self, actions):
		self.actions = actions

	def pickAction(self, reward, obs):
		return self.actions[np.random.randint(0, len(self.actions))]

# def preprocess(observation):
# 	observation = cv2.cvtColor(cv2.resize(observation, (BrainDQN.MAP_WIDTH, BrainDQN.MAP_WIDTH)), cv2.COLOR_BGR2GRAY)
# 	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
# 	return np.reshape(observation,(BrainDQN.MAP_WIDTH,BrainDQN.MAP_WIDTH,1))

def preprocess(observation, map_width, map_height):
	observation = cv2.cvtColor(cv2.resize(observation, (map_height, map_width)), cv2.COLOR_BGR2GRAY)
	# ret, observation = cv2.threshold(observation, 150, 255, cv2.THRESH_BINARY)
	return observation
	# return np.reshape(observation, (map_width, map_height, 1))

def main():
	# Step 1: init BrainDQN
	# actions = 2
	# brain = BrainDQN(actions)
	# # Step 2: init Flappy Bird Game
	# flappyBird = game.GameState()
	# # Step 3: play game
	# # Step 3.1: obtain init state
	# action0 = np.array([1,0])  # do nothing
	# observation0, reward0, terminal = flappyBird.frame_step(action0)
	# observation0 = cv2.cvtColor(cv2.resize(observation0, (BrainDQN.MAP_WIDTH, BrainDQN.MAP_WIDTH)), cv2.COLOR_BGR2GRAY)
	# ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	# brain.setInitState(observation0)

	game = MyGame()
	p = PLE(game, fps=30, display_screen=False, force_fps=True, frame_skip=1)
	p.init()

	nb_frames = 0
	reward = 0.0

	next_obs = p.getScreenRGB()
	map_width = int(next_obs.shape[0] / 2)
	map_height = int(next_obs.shape[1] / 2)

	next_obs = preprocess(next_obs, map_width, map_height)

	actions = p.getActionSet()
	myAgent = BrainDQN(actions, map_width, map_height)
	myAgent.setInitState(next_obs)

	while True:
		obs = next_obs
		action, brain_action = myAgent.getAction()
		reward = p.act(action)
		if reward != 0:
			print ('reward', reward)

		if reward == 0:
			reward = 0.1

		next_obs = p.getScreenRGB()
		next_obs = preprocess(next_obs, map_width, map_height)
		if p.game_over() or nb_frames > 5000: #check if the game is over
			myAgent.setPerception(next_obs, brain_action, reward, True)
			nb_frames = 0
			p.reset_game()
		else:
			myAgent.setPerception(next_obs, brain_action, reward, False)

		nb_frames += 1

if __name__ == '__main__':
	main()