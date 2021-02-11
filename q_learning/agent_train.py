import math
import gym
import numpy as np
import matplotlib.pyplot as plt


class QBrain:
    def __init__(self, env='', max_epsilon=1.0, min_epsilon=0.1, epsilon_decay_rate=0.01, alpha=0.01, episodes=1000,
                 gamma=0.9):
        self.env = gym.make(env)
        self.alpha = alpha
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon_decay_rate = epsilon_decay_rate
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.all_episodes_reward = []
        self.all_episodes_error = []

    # ---Choose actions with respect to decayed epsilon greedy policy
    def choose_epsilon_greedy_action(self, state):
        # --with epsilon probability choose random action otherwise choose the best action
        rnd = np.random.random()
        if rnd >= self.max_epsilon:
            return self.find_best_action_value(state)
        else:
            return self.env.action_space.sample()

    # --returns the action with highest action value
    def find_best_action_value(self, state):
        return np.argmax(self.q[state, :])

    # --epsilon decay schedule
    def epsilon_decay(self, episode):
        self.max_epsilon = max(self.min_epsilon,
                               min(self.max_epsilon, 1.0 - math.log10((episode + 1) * self.epsilon_decay_rate)))
        return self.max_epsilon

    # --start exploring environment
    def run(self):
        for episode in range(self.episodes):
            # --at the starting point of each episode we reset environment which fall us into a random state
            current_state = self.env.reset()
            time_step = 0
            current_episode_rewards = 0
            current_td_error = 0
            # --terminate defines if we are in termination state
            terminate = False
            while not terminate:
                # --from the current state in env try to choose an action with respect to
                # --epsilon greed policy, then take chosen action
                current_action = self.choose_epsilon_greedy_action(current_state)
                next_state, reward, terminate, _ = self.env.step(current_action)
                # --due to Q-learning method, bootstrap one step forward and then choose the best action value to update
                # --your current value (the beauty of this method is you update a guess by a guess! and it works :) )
                td_error = reward + (
                        (self.gamma * np.max(self.q[next_state, :])) - self.q[current_state, current_action])
                self.q[current_state, current_action] = self.q[current_state, current_action] + self.alpha * td_error
                current_state = next_state
                current_episode_rewards += reward
                current_td_error += td_error
                time_step += 1
            self.epsilon_decay(episode)
            self.all_episodes_reward.append(current_episode_rewards)
            self.all_episodes_error.append(current_td_error)

        print(self.q)
        rewards_per_thousand_episodes = np.split(np.array(self.all_episodes_reward), self.episodes / 1000)
        td_error_per_thousand_episodes = np.split(np.array(self.all_episodes_error), self.episodes / 1000)
        count = 1000
        i, j1, j2 = [], [], []
        print("Average reward per episode")
        for r, m in zip(rewards_per_thousand_episodes, td_error_per_thousand_episodes):
            print(count, ":", str(sum(r / 1000)))
            j1.append(sum(r / 1000))
            j2.append(sum(m / 1000))
            count += 1000
            i.append(count)

        # --plot total rewards and errors you got each 1000 episode
        plt.plot(i, j1, label='Reward')
        plt.plot(i, j2, label='TD Error')
        plt.xlabel('Episodes')
        plt.legend()
        plt.show()

        self.env.close()
        return self.q
