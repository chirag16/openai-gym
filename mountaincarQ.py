'''
OpenAI Gym - MountainCar-v0
Q - Learning
'''
import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('MountainCar-v0')

# Q-table
NUM_DISCRETIZATIONS = 16
NUM_ACTIONS = env.action_space.n
Q = np.zeros((NUM_DISCRETIZATIONS, NUM_DISCRETIZATIONS, NUM_ACTIONS))

# Hyperparams
eps = 8e-1
MIN_EPSILON = 1e-2
EPSILON_DECAY_RATE = 999e-3
alpha = 1
MIN_ALPHA = 1e-2
ALPHA_DECAY_RATE = 999e-3
GAMMA = 7e-1

# Helper function to convert observation to discreet state
MIN_OBSERVATION, MAX_OBSERVATION = [-1.2, -0.07], [0.6, 0.07]
def obs_to_state(observation):
	global Q

	state = np.zeros(observation.shape).astype('int')
	for i in range(len(observation)):
		observation[i] = min(max(observation[i], MIN_OBSERVATION[i]), MAX_OBSERVATION[i])
		state[i] = int((Q.shape[i] - 1) * (observation[i] - MIN_OBSERVATION[i]) / (MAX_OBSERVATION[i] - MIN_OBSERVATION[i]))

	return state[0], state[1]


# Train
NUM_EPISODES = 10000
scores = []
for ep in range(NUM_EPISODES):
	observation = env.reset()
	done = False
	score = 0

	while not done:
		# Display the environment
		if ep > 9950:
			env.render()

		# Get the state
		state = obs_to_state(observation)

		# Choose an action - Epsilon Greedy
		if np.random.rand() > eps:
			action = np.argmax(Q[state])
		else:
			action = env.action_space.sample()

		# perform that action
		observation, reward, done, _ = env.step(action)

		# Get the new state
		new_state = obs_to_state(observation)

		# Update the Q-table
		Q[state][action] += alpha * (reward + GAMMA * np.max(Q[new_state]) - Q[state][action])

		# Update score
		score += reward

	scores.append(score)

	# Decay eps
	eps *= EPSILON_DECAY_RATE
	eps = max(MIN_EPSILON, eps)

	# Decay alpha
	alpha *= ALPHA_DECAY_RATE
	alpha = max(MIN_ALPHA, alpha)

	print 'Episode:\t', ep, '\tScore:\t', score, '\teps:\t', eps, '\talpha:\t', alpha

plt.plot(range(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()

env.close()