import random
import os
from pathlib import Path
import numpy as np
import torch
import argparse

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import Agent_DQN  # Replace DummyAgent with your agent implementation
from env_hiv import HIVPatient


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    file = Path("score.txt")
    if not file.is_file():
        seed_everything(seed=42)
        # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
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
        env = HIVPatient()
        model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.nb_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(args.nb_neurons, args.nb_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(args.nb_neurons, env.action_space.n),
        )
        agent = Agent_DQN(config=config, env=env, model=model)
        agent.load()
        # Evaluate agent and write score.
        score_agent: float = evaluate_HIV(agent=agent, nb_episode=5)
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=20)
        with open(file="score.txt", mode="w") as f:
            f.write(f"{score_agent}\n{score_agent_dr}")
