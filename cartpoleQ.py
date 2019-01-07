'''
OpenAI Gym - CartPole-v0
Q - Learning
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('CartPole-v0')

# The Q-table
NUM_DISCRETIZATIONS = 16
NUM_ACTIONS = env.action_space.n
Q = np.zeros((NUM_DISCRETIZATIONS, NUM_DISCRETIZATIONS, NUM_ACTIONS)) #+ 1. # Optimistic initial values

# Hyperparams
eps = 8e-1
EPSILON_DECAY_RATE = 999e-3
MIN_EPS = 1e-2 
alpha = 1
ALPHA_DECAY_RATE = 999e-3
MIN_ALPHA = 1e-2
GAMMA = 9e-1

# Helper function to convert observation from the environment into a discreet state
THETA_ID, THETA_DOT_ID = 2, 3
MIN_THETA, MAX_THETA = -0.20, 0.20 # env.observation_space.low[THETA_ID], env.observation_space.high[THETA_ID]
MIN_THETA_DOT, MAX_THETA_DOT = -2.0, 2.0 # env.observation_space.low[THETA_DOT_ID], env.observation_space.high[THETA_DOT_ID]
def obs_to_state(observation):
	global THETA_ID, THETA_DOT_ID, Q, MIN_THETA, MAX_THETA, MIN_THETA_DOT, MAX_THETA_DOT
	theta = observation[THETA_ID]
	theta_dot = observation[THETA_DOT_ID]

	# Bounding theta and theta_dot
	theta = min(max(MIN_THETA, theta), MAX_THETA)
	theta_dot = min(max(MIN_THETA_DOT, theta_dot), MAX_THETA_DOT)

	# Discretization
	theta = int((Q.shape[0] - 1) * (theta - MIN_THETA) / (MAX_THETA - MIN_THETA))
	theta_dot = int((Q.shape[1] - 1) * (theta_dot - MIN_THETA_DOT) / (MAX_THETA_DOT - MIN_THETA_DOT))

	return theta, theta_dot

# Train
NUM_EPISODES = 10000
scores = []
for ep in range(NUM_EPISODES):
	# Reset the environment
	observation = env.reset()
	done = False
	score = 0

	while not done:
		# Display the environment
		if ep > 5000:
			env.render()

		# Get the state
		state = obs_to_state(observation)

		# Choose an aciton - Epsilon Greedy
		if np.random.rand() > eps:
			action = np.argmax(Q[state])
		else:
			action = env.action_space.sample()

		# Perform the chosen action
		observation, reward, done, _ = env.step(action)

		# Get the new state
		new_state = obs_to_state(observation)

		# Update the Q-table
		Q[state][action] += alpha * (reward + GAMMA * np.max(Q[new_state]) - Q[state][action])

		# Increase score
		score += reward

	scores.append(score)

	# Decay eps
	eps *= EPSILON_DECAY_RATE
	eps = max(MIN_EPS, eps)

	# Decay alpha
	alpha *= ALPHA_DECAY_RATE
	alpha = max(MIN_ALPHA, alpha)

	print 'Episode:\t', ep, '\tScore:\t', score, '\teps:\t', eps, '\talpha:\t', alpha

print Q
plt.plot(range(len(scores)), scores)
plt.show()

env.close()