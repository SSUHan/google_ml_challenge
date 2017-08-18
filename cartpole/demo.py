import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter


LR = 1e-3
goal_steps = 500
score_requirement = 50
initial_games = 200

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
env.reset()

def initial_population():
	# [OBS, MOVES]
	training_data = []
	# all scores
	scores = []
	# just the scores that met our threshold
	accpeted_scores = []

	for _ in range(initial_games):
		score = 0
		# moves specifically from this environment:
		game_memory = []
		# previous observation that we saw
		prev_observation = []
		# for each frame in 200
		for _ in range(goal_steps):
			# choose random action (0 or 1)
			action = random.randrange(0,2)
			# do it!
			observation, reward, done, info = env.step(action)
			
			# notice that the observation is returned FROM the action
			# so we'll store the previous observation here, pairing
			# the prev observation to the action we'll take.
			if len(prev_observation) > 0 :
				game_memory.append([prev_observation, action])
			prev_observation = observation
			score += reward
			if done:
				break

		if score >= score_requirement:
			print("Over requirement score!!")
			accpeted_scores.append(score)
			for data in game_memory:
				# convert ohe-hot (this is output layer for neural network)
				if data[1] == 1:
					output = [0, 1]
				elif data[1] == 0:
					output = [1, 0]

				# save our training data
				training_data.append([data[0], output])

		env.reset()
		scores.append(score)

	print("training data len : ", len(training_data))
	np_training_data = np.array(training_data)
	np.save('saved.npy', np_training_data)

	print('Average accepted score:', mean(accpeted_scores))
	print('Median score for accepted scores:',median(accpeted_scores))
	print(Counter(accpeted_scores))

	return training_data


def neural_network_model(input_size):
	net = input_data(shape=[None, input_size, 1], name='input')
	
	net = fully_connected(net, 128, activation='relu')
	net = dropout(net, 0.8)

	net = fully_connected(net, 256, activation='relu')
	net = dropout(net, 0.8)

	net = fully_connected(net, 512, activation='relu')
	net = dropout(net, 0.8)

	net = fully_connected(net, 256, activation='relu')
	net = dropout(net, 0.8)

	net = fully_connected(net, 128, activation='relu')
	net = dropout(net, 0.8)

	net = fully_connected(net, 2, activation='softmax')
	net = regression(net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(net, tensorboard_dir='log')

def train_model(training_data, model=False):
	print(training_data)
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	print(X)
	pass

if __name__ == '__main__':
	initial_population()
	pass
# for each_episode in range(500):
# 	observation = env.reset()
# 	for t in range(100):
# 		env.render()
# 		action = env.action_space.sample()
# 		observation, reward, done, info = env.step(action)
# 		print("Index : {},\tObservation : {},\tReword : {}".format(t, observation, reward))
# 		if done:
# 			print("Episode finished after {} timesteps".format(t+1))