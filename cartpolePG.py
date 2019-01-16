'''
Policy Gradients for CartPole-v0
'''
import gym
import numpy as np

# Create the environment
env = gym.make('CartPole-v0')

# Hyperparameters
n_input = 4
n_1 = 3
n_2 = 1
ALPHA = 5e-1
GAMMA = 9e-1

# Define the Parameters
W_1 = (np.random.rand(n_1, n_input) - 0.5) / 10
b_1 = (np.random.rand(n_1, 1) - 0.5) / 10
W_2 = (np.random.rand(n_2, n_1) - 0.5) / 10
b_2 = (np.random.rand(n_2, 1) - 0.5) / 10

# Display initial weights
print 'Initial parameters'
print '-' * 50
print 'W_1'
print '-' * 50
print W_1
print '-' * 50
print 'b_1'
print '-' * 50
print b_1
print '-' * 50
print 'W_2'
print '-' * 50
print W_2
print '-' * 50
print 'b_2'
print '-' * 50
print b_2

# Sigmoid function
def sigmoid(z):
	return 1. / (1. + np.exp(-z))

# Forward Propagation
def forward_prop(x, W_1, b_1, W_2, b_2):
	z_1 = np.matmul(W_1, x) + b_1
	a_1 = sigmoid(z_1)
	z_2 = np.matmul(W_2, a_1) + b_2
	a_2 = sigmoid(z_2)

	return z_1, a_1, z_2, a_2			# Return stuff for back prop and the prediction (a_2)

# Action selection
def pick_action(x, W_1, b_1, W_2, b_2):
	z_1, a_1, z_2, prediction = forward_prop(x, W_1, b_1, W_2, b_2)
	if prediction[0][0] >= np.random.uniform(): return z_1, a_1, z_2, prediction, 1
	return z_1, a_1, z_2, prediction, 0

# Run episodes and train
NUM_EPISODES = 5000
REQUIRED_REWARD = 190
success = 0

for ep in range(NUM_EPISODES):
	# Reset the environment
	observation = env.reset()
	done = False
	total_reward = 0
	final_reward = 0
	states, predictions, actions, a_1s = [], [], [], []

	# Run the episode
	while not done:
		# Render the environment
		if ep > NUM_EPISODES - 50:
			env.render()

		# Choose an action
		observation = np.array(observation)
		observation = np.expand_dims(observation, axis = 1) 	# To convert observation to a column vector of shape = (n_input, 1)
		z_1, a_1, z_2, prediction, action = pick_action(observation, W_1, b_1, W_2, b_2)

		# Store the current state and action prediction
		states.append(observation)
		predictions.append(prediction) 		# a_2s
		actions.append(action)
		a_1s.append(a_1)					# a_1s

		# Execute the action
		observation, immediate_reward, done, _ = env.step(action)

		# Accumulate the rewards received
		total_reward += immediate_reward

	# Print the score
	print '*' * 50
	print 'Episode:\t', ep, 'Score:\t', total_reward, '#Success:\t', success

	# Preprocess the data
	states = np.array(states)
	actions = np.array(actions)

	# Compute the fake targets
	if total_reward >= REQUIRED_REWARD:
		final_reward = 1
		success += 1
	else: final_reward = -1

	# Batch Gradient Descent
	W_1_delta = np.zeros(W_1.shape)
	b_1_delta = np.zeros(b_1.shape)
	W_2_delta = np.zeros(W_2.shape)
	b_2_delta = np.zeros(b_2.shape)

	# print W_1.shape, b_1.shape, W_2.shape, b_2.shape
	discount_factor = GAMMA ** (len(states) - 1)
	for i in range(len(states)):
		x = states[i].copy()				# Fetch the ith entry from the data	

		y = actions[i].copy()				# Fetch the ith target
		y = np.expand_dims(y, axis = 1)		# To convert y to a column vector of shape = (n_2, 1)

		a_2 = predictions[i]				# Fetch the activation of the second layer for the ith step
		a_1 = a_1s[i]						# Fetch the activation of the second layer for the ith step

		W_1_delta += np.matmul(np.multiply(W_2.T, np.multiply(a_1, 1. - a_1)), np.multiply(a_2 - y, x.T)) * final_reward * discount_factor
		b_1_delta += np.matmul(np.multiply(W_2.T, np.multiply(a_1, 1. - a_1)), a_2 - y) * final_reward * discount_factor
		W_2_delta += np.matmul(a_2 - y, a_1.T) * final_reward * discount_factor
		b_2_delta += (a_2 - y) * final_reward * discount_factor
		discount_factor /= GAMMA

	# Simultaneous update
	W_1 -= ALPHA * W_1_delta / len(states)
	b_1 -= ALPHA * b_1_delta / len(states)
	W_2 -= ALPHA * W_2_delta / len(states)
	b_2 -= ALPHA * b_2_delta / len(states)

# Display final parameters
print 'Tuned parameters'
print '-' * 50
print 'W_1'
print '-' * 50
print W_1
print '-' * 50
print 'b_1'
print '-' * 50
print b_1
print '-' * 50
print 'W_2'
print '-' * 50
print W_2
print '-' * 50
print 'b_2'
print '-' * 50
print b_2

# Close the environment
env.close()
