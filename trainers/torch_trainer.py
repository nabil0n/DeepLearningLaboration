import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import ale_py
import random
from collections import namedtuple, deque
import csv
from datetime import datetime
import os

Transition = namedtuple('Transition', 
    ('state', 'action', 'reward', 'next_state', 'done'))

gym.register_envs(ale_py)

class TorchTrainer:
    def __init__(self,
                num_frames=4, 
                num_actions=6, 
                num_episodes=20000, 
                num_steps=10000, 
                batch_size=32, 
                lr=0.0001,
                gamma=0.99, 
                epsilon=1.0,
                epsilon_min=0.1,
                epsilon_max=1.0,
                target_update_freq=1000
            ):
        self.env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
        self.env = AtariPreprocessing(self.env)
        self.env = FrameStack(self.env, num_frames)
        self.env = gym.wrappers.RecordVideo(self.env, "videos/torch", episode_trigger=lambda episode: episode % 100 == 0, disable_logger=True)
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_interval = (epsilon_max - epsilon_min)
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self.create_network().to(self.device)
        self.target_net = self.create_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = []
        self.steps_done = 0
        
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_history = []
        self.avg_rewards = deque(maxlen=100)
        self.max_rewards = deque(maxlen=100)
        self.start_time = datetime.now().timestamp()

    def create_network(self):
        return nn.Sequential(
            nn.Conv2d(self.num_frames, 32, kernel_size=8, stride=4),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # Flatten to shape: (batch_size, 3136)
            nn.Linear(64 * 7 * 7, 512),  # Input size: 3136
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)  # Output: (batch_size, num_actions)
        )

    def select_action(self, state):
        state_tensor = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(0)
        
        if np.random.rand() < self.epsilon:
            return torch.tensor([[np.random.randint(self.num_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state_tensor).argmax(dim=1).view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.sample_memory()
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(np.array(batch.state), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long).view(-1, 1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(batch.next_state), device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.bool)
    
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[~done_batch] = self.target_net(next_state_batch[~done_batch]).max(1)[0].detach()
        expected_state_action_values = reward_batch + self.gamma * next_state_values
    
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
        
    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)
    
    def train(self):
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = None
            
            for step in range(self.num_steps):
                self.steps_done += 1
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action.item())
                
                self.memory.append(Transition(state, action.item(), reward, next_state, done))
                
                state = next_state
                episode_reward += reward
                
                loss = self.optimize_model()
                if loss is not None:
                    episode_loss = loss.item()

                if done or truncated:
                    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
                    self.max_rewards.append(np.max(episode_reward))
                    if episode % 20 == 0:
                        self.log_episode_stats(episode, episode_reward, episode_loss)
                        torch.save(self.policy_net.state_dict(), f"../models/torch/space_invaders_qmodel_{episode}.pt")
                    break
                
                if self.steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.epsilon = self.epsilon_min + self.epsilon_interval * np.exp(-1. * self.steps_done / 10000)
        
        self.env.close()
        
    def log_episode_stats(self, episode, episode_reward, episode_loss):
        self.episode_rewards.append(episode_reward)
        self.avg_rewards.append(episode_reward)
        self.epsilon_history.append(self.epsilon)
        
        avg_100_episodes = np.mean(list(self.avg_rewards))
        
        log_file = f"../logs/torch/modelstats_{datetime.now():%d-%m}.csv"
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(log_file).st_size == 0:
                writer.writerow([
                    "episode",
                    "reward",
                    "running_reward",
                    "max_reward",
                    "epsilon",
                    "loss",
                    "time_since_start"
                ])
            writer.writerow([
                episode,
                episode_reward,
                avg_100_episodes,
                np.max(self.max_rewards),
                self.epsilon,
                episode_loss,
                (datetime.now().timestamp() - self.start_time)
            ])
        
        
if __name__ == "__main__":
    trainer = TorchTrainer()
    trainer.train()