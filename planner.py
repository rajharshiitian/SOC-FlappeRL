import argparse
import numpy as np
from scipy.optimize import linprog


def h_poly_iter(nStates, nActions, transitions, rewards, gamma_factor):
    values = np.zeros(nStates)
    policy = np.zeros(nStates, dtype=int)

    while True:

        while True:
            prev_values = values.copy()

            for state in range(nStates):
                # print(state)
                action = policy[state]

                value = 0.0
                for next_state in range(nStates):
                    value += transitions[state, action, next_state] * (
                        rewards[state, action, next_state] + gamma_factor * prev_values[next_state])

                values[state] = value

            if np.max(np.abs(values - prev_values)) < 1e-8:
                break

        policy_stable = True

        for state in range(nStates):
            old_action = policy[state]
            q_values = np.zeros(nActions)

            for action in range(nActions):
                q_value = 0.0

                for next_state in range(nStates):
                    q_value += transitions[state, action, next_state] * (
                        rewards[state, action, next_state] + gamma_factor * values[next_state])

                q_values[action] = q_value

            best_action = np.argmax(q_values)
            policy[state] = best_action

            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break
    return values, policy


def val_iter(nStates, nActions, transitions, rewards, gamma_factor):
    values = np.zeros(nStates)
    policy = np.zeros(nStates, dtype=int)

    while True:
        prev_values = values.copy()

        for state in range(nStates):
            q_values = np.zeros(nActions)

            for action in range(nActions):
                q_value = 0.0

                for next_state in range(nStates):
                    q_value += transitions[state, action, next_state] * (
                        rewards[state, action, next_state] + gamma_factor * prev_values[next_state])

                q_values[action] = q_value

            best_action = np.argmax(q_values)
            values[state] = q_values[best_action]
            policy[state] = best_action

        if np.max(np.abs(values - prev_values)) < 1e-9:
            break

    return values, policy


#def L_programming(nStates, nActions, transitions, rewards, gamma_factor):
    c = np.zeros(nStates)
    A_ub = np.zeros((nStates * nActions, nStates))
    b_ub = np.zeros(nStates * nActions)
    bounds = [(None, None)] * nStates

    for state in range(nStates):
        c[state] = -1.0

        for action in range(nActions):
            b_ub[state * nActions + action] = 0.0

            for next_state in range(nStates):
                A_ub[state * nActions + action,
                     next_state] = transitions[state, action, next_state]
                b_ub[state * nActions + action] += transitions[state, action, next_state] * \
                    (rewards[state, action, next_state] +
                     gamma_factor * 0.0)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    values = result.x
    policy = np.zeros(nStates, dtype=int)

    for state in range(nStates):
        q_values = np.zeros(nActions)

        for action in range(nActions):
            q_value = 0.0

            for next_state in range(nStates):
                q_value += transitions[state, action, next_state] * (
                    rewards[state, action, next_state] + gamma_factor * values[next_state])

            q_values[action] = q_value

        best_action = np.argmax(q_values)
        policy[state] = best_action

    return values, policy


def mdp_read(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    L_arr = lines

    nStates = int(lines[0].split()[1])
    nActions = int(lines[1].split()[1])
    start_state = int(lines[2].split()[1])
    end_states = list(map(int, lines[3].split()[1:]))

    TP_Matrix = np.zeros((nStates, nActions, nStates))
    rewards = np.zeros((nStates, nActions, nStates))

    for data in lines[4:-2]:
        impvalues = data.split()
        s1 = int(impvalues[1])
        ac = int(impvalues[2])
        s2 = int(impvalues[3])
        rewards[s1, ac, s2] = eval(impvalues[4])
        TP_Matrix[s1, ac, s2] = eval(impvalues[5])

    mdptype = lines[-2].split()[1]
    gamma = float(lines[-1].split()[1])

    return nStates, nActions, start_state, end_states, TP_Matrix, rewards, mdptype, gamma


def results(values, policy):
    for state in range(len(values)):
        value = values[state]
        action = policy[state]
        print(f"{value:.6f}   {action}")


def cust_arg():
    # Read command-line arguments
    cla = argparse.ArgumentParser()
    cla.add_argument("--mdp", help="Path of MDP file")
    cla.add_argument(
        "--algorithm", help=" either of (vi, hpi or lp)")
    args = cla.parse_args()

    mdp_file = args.mdp
    algorithm = args.algorithm

    # Read MDP file
    nStates, nActions, start_state, end_states, transitions, rewards, mdptype, gamma = mdp_read(
        mdp_file)

    if algorithm == "hpi":
        values, policy = h_poly_iter(
            nStates, nActions, transitions, rewards, gamma)
    elif algorithm == "vi":
        values, policy = val_iter(
            nStates, nActions, transitions, rewards, gamma)
    elif algorithm == "lp":
        values, policy = L_programming(
            nStates, nActions, transitions, rewards, gamma)
    else:
        print("Incorrect command!, choose either of 'vi', 'hpi', or 'lp'.")
        return

    # optimal value function and policy
    results(values, policy)


if __name__ == "__main__":
    cust_arg()
