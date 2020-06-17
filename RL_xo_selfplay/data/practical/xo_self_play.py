import numpy as np
from math import inf as infinity
import itertools
import random
import time
import matplotlib.pyplot as plt

def flip_d(arr):
    new = []
    for j in range(len(arr[0])):
        new_array = []
        new_array.append(arr[0][j])
        new_array.append(arr[1][j])
        new_array.append(arr[2][j])
        new.append(new_array)
    return new

def flip_h(arr):
    new = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
    for i in [0,1,2]:
        new[i][1] = arr[i][1]
        new[i][0] = arr[i][2]
    return new


def flip_v(arr):
    new = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
    for i in [0,1,2]:
        new[1][i] = arr[1][i]
        new[0][i] = arr[2][i]
    return new

def flip_d2(arr):
    new = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
    new[0][0] = arr[2][2]
    new[0][1] = arr[1][2]
    new[0][2] = arr[0][2]
    new[1][0] = arr[2][1]
    new[1][1] = arr[1][1]
    new[1][2] = arr[0][1]
    new[2][0] = arr[2][0]
    new[2][1] = arr[1][0]
    new[2][2] = arr[0][0]
    return new

def flip_90(A):

    N = len(A[0])
    for i in range(N // 2):
        for j in range(i, N - i - 1):
            temp = A[i][j]
            A[i][j] = A[N - 1 - j][i]
            A[N - 1 - j][i] = A[N - 1 - i][N - 1 - j]
            A[N - 1 - i][N - 1 - j] = A[j][N - 1 - i]
            A[j][N - 1 - i] = temp
    return A

game_state = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
players = ['X','O']

def play_move(state, player, block_num):
    if state[int((block_num-1)/3)][(block_num-1)%3] is ' ':
        state[int((block_num-1)/3)][(block_num-1)%3] = player
    else:
        block_num = int(input("Block is not empty, ya blockhead! Choose again: "))
        play_move(state, player, block_num)
    
def copy_game_state(state):
    new_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
    for i in range(3):
        for j in range(3):
            new_state[i][j] = state[i][j]
    return new_state
    
def check_current_state(game_state):    
    # Check horizontals
    if (game_state[0][0] == game_state[0][1] and game_state[0][1] == game_state[0][2] and game_state[0][0] is not ' '):
        return game_state[0][0], "Done"
    if (game_state[1][0] == game_state[1][1] and game_state[1][1] == game_state[1][2] and game_state[1][0] is not ' '):
        return game_state[1][0], "Done"
    if (game_state[2][0] == game_state[2][1] and game_state[2][1] == game_state[2][2] and game_state[2][0] is not ' '):
        return game_state[2][0], "Done"
    
    # Check verticals
    if (game_state[0][0] == game_state[1][0] and game_state[1][0] == game_state[2][0] and game_state[0][0] is not ' '):
        return game_state[0][0], "Done"
    if (game_state[0][1] == game_state[1][1] and game_state[1][1] == game_state[2][1] and game_state[0][1] is not ' '):
        return game_state[0][1], "Done"
    if (game_state[0][2] == game_state[1][2] and game_state[1][2] == game_state[2][2] and game_state[0][2] is not ' '):
        return game_state[0][2], "Done"
    
    # Check diagonals
    if (game_state[0][0] == game_state[1][1] and game_state[1][1] == game_state[2][2] and game_state[0][0] is not ' '):
        return game_state[1][1], "Done"
    if (game_state[2][0] == game_state[1][1] and game_state[1][1] == game_state[0][2] and game_state[2][0] is not ' '):
        return game_state[1][1], "Done"
    
    # Check if draw
    draw_flag = 0
    for i in range(3):
        for j in range(3):
            if game_state[i][j] is ' ':
                draw_flag = 1
    if draw_flag is 0:
        return None, "Draw"
    
    return None, "Not Done"

def print_board(game_state):
    print('----------------')
    print('| ' + str(game_state[0][0]) + ' || ' + str(game_state[0][1]) + ' || ' + str(game_state[0][2]) + ' |')
    print('----------------')
    print('| ' + str(game_state[1][0]) + ' || ' + str(game_state[1][1]) + ' || ' + str(game_state[1][2]) + ' |')
    print('----------------')
    print('| ' + str(game_state[2][0]) + ' || ' + str(game_state[2][1]) + ' || ' + str(game_state[2][2]) + ' |')
    print('----------------')
    
  
# Initialize state values
player = ['X','O',' ']
states_dict = {}
all_possible_states = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player, repeat = 9)]
n_states = len(all_possible_states) # 2 players, 9 spaces
n_actions = 9   # 9 spaces
state_values_for_AI_O = np.full((n_states),0.0)
state_values_for_AI_X = np.full((n_states),0.0)
# print("n_states = %i \nn_actions = %i"%(n_states, n_actions))

# State values for AI 'O'
for i in range(n_states):
    states_dict[i] = all_possible_states[i]
    winner, _ = check_current_state(states_dict[i])
    if winner == 'O':   # AI won
        state_values_for_AI_O[i] = 1
    elif winner == 'X':   # AI lost
        state_values_for_AI_O[i] = -1
        
# State values for AI 'X'       
for i in range(n_states):
    winner, _ = check_current_state(states_dict[i])
    if winner == 'O':   # AI lost
        state_values_for_AI_X[i] = -1
    elif winner == 'X':   # AI won
        state_values_for_AI_X[i] = 1

def update_state_value_O(curr_state_idx, next_state_idx, learning_rate):
    new_value = state_values_for_AI_O[curr_state_idx] + learning_rate*(state_values_for_AI_O[next_state_idx]  - state_values_for_AI_O[curr_state_idx])
    state_values_for_AI_O[curr_state_idx] = new_value
    
def update_state_value_X(curr_state_idx, next_state_idx, learning_rate):
    new_value = state_values_for_AI_X[curr_state_idx] + learning_rate*(state_values_for_AI_X[next_state_idx]  - state_values_for_AI_X[curr_state_idx])
    state_values_for_AI_X[curr_state_idx] = new_value

def getBestMove(state, player, epsilon):
    '''
    Reinforcement Learning Algorithm
    '''    
    moves = []
    curr_state_values = []
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if state[i][j] is ' ':
                empty_cells.append(i*3 + (j+1))
    
    for empty_cell in empty_cells:
        moves.append(empty_cell)
        new_state = copy_game_state(state)
        play_move(new_state, player, empty_cell)
        next_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_state)]
        if player == 'X':
            curr_state_values.append(state_values_for_AI_X[next_state_idx])
        else:
            curr_state_values.append(state_values_for_AI_O[next_state_idx])
        
    # print('Possible moves = ' + str(moves))
    # print('Move values = ' + str(curr_state_values))    
    best_move_idx = np.argmax(curr_state_values)
    
    if np.random.uniform(0,1) <= epsilon:       # Exploration
        best_move = random.choice(empty_cells)
        # print('Agent decides to explore! Takes action = ' + str(best_move))
        epsilon *= 0.99
    else:   #Exploitation
        best_move = moves[best_move_idx]
        # print('Agent decides to exploit! Takes action = ' + str(best_move))
    return best_move

# PLaying

#LOAD TRAINED STATE VALUES
# state_values_for_AI_X = np.loadtxt('trained_state_values_X.txt', dtype=np.float64)
# state_values_for_AI_O = np.loadtxt('trained_state_values_O.txt', dtype=np.float64)
score_X = []
score_O = []

learning_rate = 0.2
epsilon = 0.2
num_iterations = 10000
for iteration in range(num_iterations):
    game_state = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
    current_state = "Not Done"
    print("Iteration " + str(iteration) + "!")
    # print_board(game_state)
    winner = None
    current_player_idx = random.choice([0,1])
        
    while current_state == "Not Done":
        lcurr_state_idx = list(states_dict.keys())[list(states_dict.values()).index(game_state)]
        if current_player_idx == 0:     # AI_X's turn
            # print("\nAI X's turn!")
            block_choice = getBestMove(game_state, players[current_player_idx], epsilon)
            play_move(game_state ,players[current_player_idx], block_choice)
            lnew_state_idx = list(states_dict.keys())[list(states_dict.values()).index(game_state)]

        else:       # AI_O's turn
            # print("\nAI O's turn!")
            block_choice = getBestMove(game_state, players[current_player_idx], epsilon)
            play_move(game_state ,players[current_player_idx], block_choice)
            lnew_state_idx = list(states_dict.keys())[list(states_dict.values()).index(game_state)]

        # print_board(game_state)
        #print('State value = ' + str(state_values_for_AI[new_state_idx]))

        update_state_value_O(lcurr_state_idx, lnew_state_idx, learning_rate)
        update_state_value_X(lcurr_state_idx, lnew_state_idx, learning_rate)



        new_game_state = flip_90(game_state)
        curr_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        new_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]


        update_state_value_O(curr_state_idx, new_state_idx, learning_rate)
        update_state_value_X(curr_state_idx, new_state_idx, learning_rate)
        #
        new_game_state = flip_d(game_state)
        curr_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        new_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        update_state_value_O(curr_state_idx, new_state_idx, learning_rate)
        update_state_value_X(curr_state_idx, new_state_idx, learning_rate)
        #
        new_game_state = flip_d2(game_state)
        curr_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        new_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        update_state_value_O(curr_state_idx, new_state_idx, learning_rate)
        update_state_value_X(curr_state_idx, new_state_idx, learning_rate)
        #
        new_game_state = flip_h(game_state)
        curr_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        new_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        update_state_value_O(curr_state_idx, new_state_idx, learning_rate)
        update_state_value_X(curr_state_idx, new_state_idx, learning_rate)

        new_game_state = flip_v(game_state)
        curr_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        new_state_idx = list(states_dict.keys())[list(states_dict.values()).index(new_game_state)]
        update_state_value_O(curr_state_idx, new_state_idx, learning_rate)
        update_state_value_X(curr_state_idx, new_state_idx, learning_rate)
        winner, current_state = check_current_state(game_state)
        if winner is not None:
            # print(str(winner) + " won!")
            pass
        else:
            current_player_idx = (current_player_idx + 1)%2
        
        if current_state is "Draw":
            # print("Draw!")
            pass
            
        #time.sleep(1)
    if len(score_X)== 0:

            winner, state = check_current_state(game_state)
            if winner == 'X':
                score_X.append(1)
            elif winner == 'O':
                score_X.append(-1)
            else:
                score_X.append(0)
    else:
        winner, state = check_current_state(game_state)
        if winner == 'X':
            score_X.append(score_X[-1] + 1)
        elif winner == 'O':
            score_X.append(score_X[-1] - 1)
        else:
            score_X.append(score_X[-1] + 0)

    if len(score_O) == 0:
        winner, state = check_current_state(game_state)
        if winner == 'O':
            score_O.append(1)
        elif winner == 'X':
            score_O.append(-1)
        else:
            score_O.append(0)

    else:
        winner, state = check_current_state(game_state)
        if winner == 'O':
            score_O.append(score_O[-1] + 1)
        elif winner == 'X':
            score_O.append(score_O[-1] - 1)
        else:
            score_O.append(score_O[-1] + 0)

print('Training Complete!')    

# Save state values for future use
np.savetxt('trained_state_values_X.txt', state_values_for_AI_X, fmt = '%.6f')
np.savetxt('trained_state_values_O.txt', state_values_for_AI_O, fmt = '%.6f')

x = [i for i in range(1, 10001)]
plt.scatter(x, score_X)
plt.scatter(x, score_O)
plt.show()
