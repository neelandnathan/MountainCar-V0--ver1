import gym
import numpy as np 

# Make Environment
env = gym.make("MountainCar-v0")

# Setting Up Variables

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
SHOW_EVERY = 500

epsilon = 0.8
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

# Helper Functions
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

# Start Training

for episode in range(EPISODES):

    # Render Every SHOW_EVERY Episodes
    if (episode % SHOW_EVERY == 0):
        print(episode)
        render = True
    else:
        render = False

    # Get DiscreteState
    discrete_state = get_discrete_state(env.reset())

    # Begin Game

    done = False

    while not done:

        # Randomly Choosing Action Based on Epsilon
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # Step Through Game
        new_state, reward, done, _ = env.step(action)

        # New DiscreteState
        new_discrete_state = get_discrete_state(new_state)

        # RenderGame 
        if render:
            env.render()

        # Updating Q-Table
        if not done:
            max_future_q = max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            # FORMULA!
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Epsilon Decay
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()