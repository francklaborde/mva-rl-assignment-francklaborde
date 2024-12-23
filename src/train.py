from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        batch = list(map(np.array, zip(*batch)))  # Convert to numpy arrays
        return [torch.tensor(item, dtype=torch.float32, device=self.device) for item in batch]
    def __len__(self):
        return len(self.data)
    
def greedy_action(network, state):
    device = next(network.parameters()).device
    with torch.no_grad():
        Q = network(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        return torch.argmax(Q).item()
    

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class Agent_DQN:
    def __init__(self, config, env, model):
        self.env = env
        self.device = config['device']
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.max_timesteps = config['max_timesteps']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = model.to(self.device)  # Move model to device
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.verbose = config['verbose']
        self.plot = config['plot']

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            update = R + self.gamma * QYmax * (1 - D)
            QXA = self.model(X).gather(1, A.long().unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, max_episodes=200, plot=False):
        episode_return = []
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        print("Using device:", self.device)
        for episode in range(max_episodes):
            episode_cum_reward = 0
            done = False
            # There is no done=True in the env.step() function, so we need to track the number of steps
            for j in range(self.max_timesteps):              
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = greedy_action(self.model, state)
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward
                self.gradient_step()
                state = next_state if not done else env.reset()[0]
                if self.verbose:
                    print(f"Step {step}, State, : {state}, Action: {action}, Reward: {reward}")
                step += 1
            episode_return.append(episode_cum_reward)
            print(f"Episode {episode + 1}, Return: {episode_cum_reward:.2f}, Epsilon: {epsilon:.2f}")
        if plot:
            plt.plot(episode_return)
            plt.xlabel('Episode')
            plt.ylabel('Return')
            plt.show()
        return episode_return

    def act(self, observation, use_random=False):
        if use_random or np.random.rand() < self.epsilon_max:
            return np.random.randint(self.action_space.n)
        else:
            return greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_max", type=float, default=1.0)
    parser.add_argument("--epsilon_decay_period", type=int, default=1000)
    parser.add_argument("--epsilon_delay_decay", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--nb_neurons", type=int, default=24)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--max_timesteps", type=int, default=200)

    args = parser.parse_args()

    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "epsilon_min": args.epsilon_min,
        "epsilon_max": args.epsilon_max,
        "epsilon_decay_period": args.epsilon_decay_period,
        "epsilon_delay_decay": args.epsilon_delay_decay,
        "batch_size": args.batch_size,
        "verbose": args.verbose,
        "max_timesteps": args.max_timesteps,
        "plot": args.plot,
    }

    model = torch.nn.Sequential(
        torch.nn.Linear(env.observation_space.shape[0], args.nb_neurons),
        torch.nn.ReLU(),
        torch.nn.Linear(args.nb_neurons, args.nb_neurons),
        torch.nn.ReLU(),
        torch.nn.Linear(args.nb_neurons, env.action_space.n),
    )

    agent = Agent_DQN(config, env, model)
    episode_return = agent.train(200, plot=True)
    agent.save("model.pth")