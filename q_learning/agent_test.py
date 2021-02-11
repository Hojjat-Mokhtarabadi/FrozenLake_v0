import os

import gym
import numpy as np
import time
from IPython.display import clear_output


class AgentTest:
    def __init__(self, q=[], env='', episodes=5):
        self.q = q
        self.env = gym.make(env)
        self.episodes = episodes

    def choose_action(self, state):
        return np.argmax(self.q[state, :])

    def start_acting(self):
        for episode in range(self.episodes):
            print('here')
            current_state = self.env.reset()
            terminate = False
            time.sleep(1)
            while not terminate:
                clear_output(wait=True)
                os.system('cls')
                self.env.render()
                time.sleep(0.3)

                current_action = self.choose_action(current_state)
                next_state, reward, terminate, _ = self.env.step(current_action)
                current_state = next_state

                if terminate:
                    if reward == 1:
                        print("goal")
                    else:
                        print("felt in hole")
        self.env.close()
