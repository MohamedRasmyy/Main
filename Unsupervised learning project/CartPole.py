#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import division, print_function
from env import CartPole, Physics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter


def initialize_mdp_data(num_states):

    transition_counts = np.zeros((num_states, num_states, 2))
    transition_probs = np.ones((num_states, num_states, 2)) / num_states
    reward_counts = np.zeros((num_states, 2))
    reward = np.zeros(num_states)
    value = np.random.rand(num_states) * 0.1

    return {
        'transition_counts': transition_counts,
        'transition_probs': transition_probs,
        'reward_counts': reward_counts,
        'reward': reward,
        'value': value,
        'num_states': num_states,
    }

def choose_action(state, mdp_data):
    
    expect_value = mdp_data['value'].dot(mdp_data['transition_probs'][state])
    if expect_value[0] == expect_value[1]:
        return np.random.randint(2)
    else:
        return np.argmax(expect_value)

def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward): 

    mdp_data['transition_counts'][state, new_state, action] += 1

    if reward == -1:
        mdp_data['reward_counts'][new_state, 0] += 1

    mdp_data['reward_counts'][new_state, 1] += 1

    return

def update_mdp_transition_probs_reward(mdp_data):
    
    transition_counts = mdp_data['transition_counts']
    num_counts = transition_counts.sum(axis=1)
    num_states = transition_counts.shape[0]
    for i in range(num_states):
        for a in range(2):
            if  num_counts[i, a]  != 0:
                mdp_data['transition_probs'][i, :, a] = transition_counts[i, :, a] / num_counts[i, a]

    reward_counts = mdp_data['reward_counts']
    for k in range(num_states):
        sum_count = reward_counts[k, 1]
        if sum_count != 0:
            mdp_data['reward'][k] = -reward_counts[k, 0] / sum_count

    return

def update_mdp_value(mdp_data, tolerance, gamma):
   

    iters = 0
    transition_probs = mdp_data['transition_probs']

    while True:
        iters += 1

        value = mdp_data['value']
        new_value = mdp_data['reward'] + gamma * value.dot(transition_probs).max(axis=1)
        mdp_data['value'] = new_value

        if np.max(np.abs(value - new_value)) < tolerance:
            break

    return iters==1

def main(plot=True):
    # Seed the randomness of the simulation so this outputs the same thing each time
    seed = 0
    np.random.seed(seed)

    # Simulation parameters
    pause_time = 0.0001
    min_trial_length_to_start_display = 100
    display_started = min_trial_length_to_start_display == 0

    NUM_STATES = 163
    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20

    # Time cycle of the simulation
    time = 0

    # These variables perform bookkeeping (how many cycles was the pole
    # balanced for before it fell). Useful for plotting learning curves.
    time_steps_to_failure = []
    num_failures = 0
    time_at_start_of_current_trial = 0

    # You should reach convergence well before this
    max_failures = 500

    # Initialize a cart pole
    cart_pole = CartPole(Physics())

    # Starting `state_tuple` is (0, 0, 0, 0)
    # x, x_dot, theta, theta_dot represents the actual continuous state vector
    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    state_tuple = (x, x_dot, theta, theta_dot)

    # `state` is the number given to this state, you only need to consider
    # this representation of the state
    state = cart_pole.get_state(state_tuple)
    # if min_trial_length_to_start_display == 0 or display_started == 1:
    #     cart_pole.show_cart(state_tuple, pause_time)

    mdp_data = initialize_mdp_data(NUM_STATES)

    # This is the criterion to end the simulation.
    # You should change it to terminate when the previous
    # 'NO_LEARNING_THRESHOLD' consecutive value function computations all
    # converged within one value function iteration. Intuitively, it seems
    # like there will be little learning after this, so end the simulation
    # here, and say the overall algorithm has converged.

    consecutive_no_learning_trials = 0
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:

        action = choose_action(state, mdp_data)

        # Get the next state by simulating the dynamics
        state_tuple = cart_pole.simulate(action, state_tuple)
        # x, x_dot, theta, theta_dot = state_tuple

        # Increment simulation time
        time = time + 1

        # Get the state number corresponding to new state vector
        new_state = cart_pole.get_state(state_tuple)
        # if display_started == 1:
        #     cart_pole.show_cart(state_tuple, pause_time)

        # reward function to use - do not change this!
        if new_state == NUM_STATES - 1:
            R = -1
        else:
            R = 0

        update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, R)

        # Recompute MDP model whenever pole falls
        # Compute the value function V for the new model
        if new_state == NUM_STATES - 1:

            update_mdp_transition_probs_reward(mdp_data)

            converged_in_one_iteration = update_mdp_value(mdp_data, TOLERANCE, GAMMA)

            if converged_in_one_iteration:
                consecutive_no_learning_trials = consecutive_no_learning_trials + 1
            else:
                consecutive_no_learning_trials = 0

        # Do NOT change this code: Controls the simulation, and handles the case
        # when the pole fell and the state must be reinitialized.
        if new_state == NUM_STATES - 1:
            num_failures += 1
            if num_failures >= max_failures:
                break
            print('[INFO] Failure number {}'.format(num_failures))
            time_steps_to_failure.append(time - time_at_start_of_current_trial)
            # time_steps_to_failure[num_failures] = time - time_at_start_of_current_trial
            time_at_start_of_current_trial = time

            if time_steps_to_failure[num_failures - 1] > min_trial_length_to_start_display:
                display_started = 1

            # Reinitialize state
            # x = 0.0
            x = -1.1 + np.random.uniform() * 2.2
            x_dot, theta, theta_dot = 0.0, 0.0, 0.0
            state_tuple = (x, x_dot, theta, theta_dot)
            state = cart_pole.get_state(state_tuple)
        else:
            state = new_state

    if plot:
        # plot the learning curve (time balanced vs. trial)
        log_tstf = np.log(np.array(time_steps_to_failure))
        plt.plot(np.arange(len(time_steps_to_failure)), log_tstf, 'k')
        window = 30
        w = np.array([1/window for _ in range(window)])
        weights = lfilter(w, 1, log_tstf)
        x = np.arange(window//2, len(log_tstf) - window//2)
        plt.plot(x, weights[window:len(log_tstf)], 'r--')
        plt.xlabel('Num failures')
        plt.ylabel('Log of num steps to failure')
        plt.title('seed = {}'.format(seed))
        plt.savefig('output/control_{}.png'.format(seed))

    return np.array(time_steps_to_failure)
    
if __name__ == '__main__':
    main()


# In[ ]:




