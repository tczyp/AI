# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np
import neat
import gym
import visualize

CONFIG = "./config"
EP_STEP = 500           # maximum episode steps
GENERATION_EP = 10      # evaluate by the minimum of 10-episode rewards
TRAINING = False         # training or testing
CHECKPOINT = 99          # test on this checkpoint

MAP_WIDTH = 5

def run():
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
	pop = neat.Population(config)

	#pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)

	# recode history
	stats = neat.StatisticsReporter()
	pop.add_reporter(stats)
	pop.add_reporter(neat.StdOutReporter(True))
	pop.add_reporter(neat.Checkpointer(5))

	pop.run(eval_genomes, 10000)       # train 10 generations

	# visualize training
	visualize.plot_stats(stats, ylog=False, view=True)
	visualize.plot_species(stats, view=True)

# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (MAP_WIDTH, MAP_WIDTH)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	cv2.imshow("img", observation)
	return np.reshape(observation,(MAP_WIDTH * MAP_WIDTH))

def eval_genomes(genomes, config):
	flappyBird = game.GameState()
	for genome_id, genome in genomes:
		run_genome(flappyBird, config, genome)

def eval_best_genomes(genomes, config):
	flappyBird = game.GameState()
	best_genome = None
	best_fitness = -99999
	for genome in iter(genomes.values()):
		if not genome.fitness:
			continue
		print (genome.fitness)
		if genome.fitness > best_fitness:
			best_fitness = genome.fitness
			best_genome = genome

	run_genome(flappyBird, config, best_genome)
	return best_genome

def run_genome(flappyBird, config, genome):
	flappyBird.reset()
	action = np.array([1, 0])
	observation, reward, terminal = flappyBird.frame_step(action)
	observation = preprocess(observation)
	observation = np.hstack((observation, observation - observation))

	net = neat.nn.FeedForwardNetwork.create(genome, config)
	ep_r = []
	#for ep in range(GENERATION_EP):  # run many episodes for the genome in case it's lucky
	accumulative_r = 0.  # stage longer to get a greater episode reward
	for t in range(EP_STEP):
		action_values = net.activate(observation)
		action_index = int(np.argmax(action_values))
		action = np.array([0, 0])
		action[action_index] = 1
		observation_, reward, terminal = flappyBird.frame_step(action)
		observation_ = preprocess(observation_)
		accumulative_r += reward
		if terminal:
			break
		#observation = observation_
		#old_ob = observation[MAP_WIDTH * MAP_WIDTH - 1:-1]
		old_ob = observation[0:MAP_WIDTH * MAP_WIDTH]
		observation = np.hstack((observation_, observation_ - old_ob))
	#ep_r.append(accumulative_r)
	#genome.fitness = np.min(ep_r) / float(EP_STEP)  # depends on the minimum episode reward
	genome.fitness = t


def evaluation():
	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
	#winner = p.run(eval_best_genomes, 1)     # find the winner in restored population
	winner = eval_best_genomes(p.population,p.config)

	# show winner net
	node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
	visualize.draw_net(p.config, winner, True, node_names=node_names)

	# net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    #
	# #flappyBird = game.GameState()
	# flappyBird = None
	# while True:
	# 	flappyBird.reset()
	# 	action = np.array([1, 0])
	# 	observation, reward, terminal = flappyBird.frame_step(action)
	# 	observation = preprocess(observation.append(observation))
	# 	while True:
	# 		action_values = net.activate(observation)
	# 		action_index = int(np.argmax(action_values))
	# 		action = np.array([0, 0])
	# 		action[action_index] = 1
	# 		observation_, reward, terminal = flappyBird.frame_step(action)
	# 		observation_ = preprocess(observation_)
	# 		observation = observation[99:-1].append(observation_)
	# 		if terminal: break

def main():
	run()
	evaluation()

if __name__ == '__main__':
	main()