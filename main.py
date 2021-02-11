from agent_train import QBrain
from agent_test import AgentTest

ENVIRONMENT = 'FrozenLake-v0'

agent = QBrain(env=ENVIRONMENT, episodes=20000, gamma=0.99, alpha=0.05)

if __name__ == "__main__":
    q = agent.run()
    agent_test = AgentTest(q=q, env=ENVIRONMENT, episodes=2)
    agent_test.start_acting()
