import numpy as np

# Define the mini maze environment
num_states = 9
num_actions = 4

# Initialize the Q-table
Q_table = np.zeros(9)
Q_table[2] = 1
Q_policy = ["+", "+", "*", "+", "+", "+", "+", "+", "+"]

# Set the learning rate (α) and exploration rate (ε)
learning_rate = 0.1
exploration_rate = 0.1
discount_factor = 0.9
count = 0
k = 10
# Perform Q-learning iterations
while "+" in Q_policy:
    print("Iteration: ", count)
    #print Q_table
    new_Q_table = np.zeros(9)
    for i in range(3):
        for j in range(3):
            print("{:.3f}".format(Q_table[3*i+j]), end=" ")
        print("\n")

    #print Q_policy
    for i in range(3):
        for j in range(3):
            print(Q_policy[3*i+j], " ", end=" ")
        print("\n")

    #learning
    if Q_policy[0] == "+":
        new_Q_table[0] += discount_factor * max(Q_table[3], Q_table[1])
        if new_Q_table[0] != 0:
            if np.argmax([Q_table[3], Q_table[1]]) == 0:
                Q_policy[0] = "↓"
            elif np.argmax([Q_table[3], Q_table[1]]) == 1:
                Q_policy[0] = "→"
    if Q_policy[1] == "+":
        new_Q_table[1] += discount_factor * max(Q_table[4], Q_table[2], Q_table[0])
        if new_Q_table[1] != 0:
            if np.argmax([Q_table[4], Q_table[2], Q_table[0]]) == 0:
                Q_policy[1] = "↓"
            elif np.argmax([Q_table[4], Q_table[2], Q_table[0]]) == 1:
                Q_policy[1] = "→"
            elif np.argmax([Q_table[4], Q_table[2], Q_table[0]]) == 2:
                Q_policy[1] = "←"
    if Q_policy[3] == "+":
        new_Q_table[3] += discount_factor * max(Q_table[6], Q_table[4], Q_table[0])
        if new_Q_table[3] != 0:
            if np.argmax([Q_table[6], Q_table[4], Q_table[0]]) == 0:
                Q_policy[3] = "↓"
            elif np.argmax([Q_table[6], Q_table[4], Q_table[0]]) == 1:
                Q_policy[3] = "→"
            elif np.argmax([Q_table[6], Q_table[4], Q_table[0]]) == 2:
                Q_policy[3] = "↑"
    if Q_policy[4] == "+":
        new_Q_table[4] += discount_factor * max(Q_table[7], Q_table[5], Q_table[3], Q_table[1])
        if new_Q_table[4] != 0:
            if np.argmax([Q_table[7], Q_table[5], Q_table[3], Q_table[1]]) == 0:
                Q_policy[4] = "↓"
            elif np.argmax([Q_table[7], Q_table[5], Q_table[3], Q_table[1]]) == 1:
                Q_policy[4] = "→"
            elif np.argmax([Q_table[7], Q_table[5], Q_table[3], Q_table[1]]) == 2:
                Q_policy[4] = "←"
            elif np.argmax([Q_table[7], Q_table[5], Q_table[3], Q_table[1]]) == 3:
                Q_policy[4] = "↑"
    if Q_policy[5] == "+":
        new_Q_table[5] += discount_factor * max(Q_table[8], Q_table[4], Q_table[2])
        if new_Q_table[5] != 0:
            if np.argmax([Q_table[8], Q_table[4], Q_table[2]]) == 0:
                Q_policy[5] = "↓"
            elif np.argmax([Q_table[8], Q_table[4], Q_table[2]]) == 1:
                Q_policy[5] = "←"
            elif np.argmax([Q_table[8], Q_table[4], Q_table[2]]) == 2:
                Q_policy[5] = "↑"
    if Q_policy[6] == "+":
        new_Q_table[6] += discount_factor * max(Q_table[7], Q_table[3])
        if new_Q_table[6] != 0:
            if np.argmax([Q_table[7], Q_table[3]]) == 0:
                Q_policy[6] = "→"
            elif np.argmax([Q_table[7], Q_table[3]]) == 1:
                Q_policy[6] = "↑"
    if Q_policy[7] == "+":
        new_Q_table[7] += discount_factor * max(Q_table[8], Q_table[6], Q_table[4])
        if new_Q_table[7] != 0:
            if np.argmax([Q_table[8], Q_table[6], Q_table[4]]) == 0:
                Q_policy[7] = "→"
            elif np.argmax([Q_table[8], Q_table[6], Q_table[4]]) == 1:
                Q_policy[7] = "←"
            elif np.argmax([Q_table[8], Q_table[6], Q_table[4]]) == 2:
                Q_policy[7] = "↑"
    if Q_policy[8] == "+":
        new_Q_table[8] += discount_factor * max(Q_table[7], Q_table[5])
        if new_Q_table[8] != 0:
            if np.argmax([Q_table[7], Q_table[5]]) == 0:
                Q_policy[8] = "←"
            elif np.argmax([Q_table[7], Q_table[5]]) == 1:
                Q_policy[8] = "↑"
    for i in range(9):
        if Q_policy[i] != "+" and Q_table[i] == 0:
            Q_table[i] = new_Q_table[i]
    count += 1

# Print the final Q-table
print("Final Q-table:")
for i in range(3):
    for j in range(3):
        print("{:.3f}".format(Q_table[3*i+j]), end=" ")
    print("\n")
print("Optimal policy:")
for i in range(3):
    for j in range(3):
        print(Q_policy[3*i+j], " ", end=" ")
    print("\n")
