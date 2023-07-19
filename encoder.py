import argparse

# possible directions
ACTIONS = ['U', 'D', 'R', 'L']


def check_valid_move(list, i, j, action):

    rows, cols = len(list), len(list[0])
    if action == 'U':
        return i > 0 and list[i - 1][j] != '1'
    elif action == 'D':
        return i < rows - 1 and list[i + 1][j] != '1'
    elif action == 'R':
        return j < cols - 1 and list[i][j + 1] != '1'
    elif action == 'L':
        return j > 0 and list[i][j - 1] != '1'
    return False


def encode_maze_as_mdp(gridfile):

    # Read the grid file and convert into a list.
    grid = []
    with open(gridfile, 'r') as file:
        for line in file:
            row = line.strip().split()
            grid.append(row)
    list = grid
    rows, cols = len(list), len(list[0])
    num_states = rows * cols
    num_actions = len(ACTIONS)

    # Initialize a MDP string
    lines_of_mdp = f"numStates {num_states}\n"
    lines_of_mdp += f"numActions {num_actions}\n"

    # appending start states to a separate array
    start_states = []
    for i in range(rows):
        for j in range(cols):
            if list[i][j] == '2':
                start_states.append(i * cols + j)
    lines_of_mdp += "start " + " ".join(str(state)
                                        for state in start_states) + "\n"

    # appending end states to a separate array
    end_states = []
    for i in range(rows):
        for j in range(cols):
            if list[i][j] == '3':
                end_states.append(i * cols + j)
    lines_of_mdp += "end " + " ".join(str(state)
                                      for state in end_states) + "\n"

    # Encode transitions and rewards to mdp strings
    for state in range(num_states):
        i, j = divmod(state, cols)
        for act_num, action in enumerate(ACTIONS):
            if check_valid_move(list, i, j, action):
                if action == 'U':
                    new_state = state - cols
                elif action == 'D':
                    new_state = state + cols
                elif action == 'R':
                    new_state = state + 1
                elif action == 'L':
                    new_state = state - 1

                new_i, new_j = divmod(new_state, cols)
                if list[i][j] in ['0', '2']:
                    if list[new_i][new_j] == '0':
                        lines_of_mdp += f"transition {state} {act_num} {new_state} -1 1.0\n"
                    if list[new_i][new_j] == '2':
                        lines_of_mdp += f"transition {state} {act_num} {new_state} -2 1.0\n"
                    if list[new_i][new_j] == '3':
                        lines_of_mdp += f"transition {state} {act_num} {new_state} 100000000000 1.0\n"

    # Add the type and gamma factor to the string
    lines_of_mdp += "mdptype episodic\n"
    # just in case any case lead to infinite loop we take gamma to be 0.95 for safety
    lines_of_mdp += "discount 0.95"

    return lines_of_mdp


def cust_arg():
    # adding functionality for reading specific command-line-arguments
    cla = argparse.ArgumentParser(description='maze to MDP')
    cla.add_argument('--grid', help=' file path')
    args = cla.parse_args()

    # convert the maze as an MDP
    lines_of_mdp = encode_maze_as_mdp(args.grid)

    # Output the MDP
    print(lines_of_mdp)
    return lines_of_mdp


if __name__ == '__main__':
    # text_file = open("mdpfile.txt", "w")
    # n = text_file.write(main())
    # text_file.close()
    cust_arg()
