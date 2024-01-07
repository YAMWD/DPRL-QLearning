import numpy as np
import matplotlib.pyplot as plt

# Initialize Q-table
num_states = 9
num_actions = 4
Q = np.full((num_states, num_actions), 0.0)
# Q[2] = 1  # set the goal state

# Set hyperparameters
alpha = 0.8  # learning rate
gamma = 0.9  # discount factor

# Define ε-greedy policy
def epsilon_greedy(state, epsilon):
    options = []
    Q_options = []
    cur_x = state // 3
    cur_y = state % 3
    if cur_x - 1 >= 0:
        options.append(0)
        Q_options.append(Q[state, 0])
    else:
        Q_options.append(-100)
    if cur_y + 1 <= 2:
        options.append(1)
        Q_options.append(Q[state, 1])
    else:
        Q_options.append(-100)
    if cur_x + 1 <= 2:
        options.append(2)
        Q_options.append(Q[state, 2])
    else:
        Q_options.append(-100)
    if cur_y - 1 >= 0:
        options.append(3)
        Q_options.append(Q[state, 3])
    else:
        Q_options.append(-100)
    
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(options)
    else:
        action = np.argmax(Q_options)
    return action

# Define function to get the next state
def get_next_state(state, action):
    if action == 0:
        next_state = state - 3
    elif action == 1:
        next_state = state + 1
    elif action == 2:
        next_state = state + 3
    elif action == 3:
        next_state = state - 1
    return next_state

# Define function to get the reward
def get_reward(state):
    if state == 2:
        reward = 1
    else:
        reward = 0
    return reward

# Q-learning algorithm
def q_learning(num_episodes, epsilon):
    steps = []
    for episode in range(num_episodes):
        state = 6  # initial state
        episode_steps = 0
        while True:
            episode_steps += 1
            action = epsilon_greedy(state, epsilon)
            next_state = get_next_state(state, action)
            reward = get_reward(state)
            
            if state == 2:
                Q[state] = 1
                steps.append(episode_steps)
                break
            else:
                Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
                state = next_state
                #print(next_state)
    
    print(Q)
    print(np.max(Q, axis = 1).reshape(3, 3))
    print("finished after average {} timesteps".format(np.mean(steps)))
    return np.mean(steps)

# Run Q-learning with different epsilon values
num_episodes = 1000
epsilons = [i*0.05 for i in range(1, 21)]
steps_per_epsilon = []
for epsilon in epsilons:
    steps = q_learning(num_episodes, epsilon)
    steps_per_epsilon.append(steps)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(epsilons, steps_per_epsilon, marker="o")
plt.title("Q-Learning: Steps for Different Epsilon Values (Learning Rate: {})".format(alpha))
plt.xlabel("ε")
plt.ylabel("Steps")
plt.legend()
plt.show()
