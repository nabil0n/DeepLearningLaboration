import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
import ale_py
from collections import deque
import csv
import os
from datetime import datetime

gym.register_envs(ale_py)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

batch_size = 32
max_steps_per_episode = 10000
max_episodes = 0
max_frames = 1e7
num_frames = 4

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

env = gym.wrappers.RecordVideo(
                env,
                video_folder=f"{parent_dir}/videos/torch",
                episode_trigger=lambda episode: episode % 100 == 0,
                disable_logger=True
            )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(num_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_dqn():
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_interval = (epsilon_max - epsilon_min)
    
    num_actions = 6

    model = DQN(num_actions).to(device)
    target_model = DQN(num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.00025)
    
    replay_buffer = ReplayBuffer(max_size=100000)
    
    epsilon_random_frames = 50000
    epsilon_greedy_frames = 1_000_000.0
    update_after_actions = 6
    update_target_network = 10000
    
    frame_count = 0
    episode_count = 0
    episode_reward_history = []
    running_reward = 0
    
    start_time = datetime.now().timestamp()
    
    while True:
        observation, _ = env.reset()
        state = np.array(observation)
        episode_reward = 0
        
        for timestep in range(1, max_steps_per_episode):
            frame_count += 1
            
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                action = np.random.choice(num_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
                    action = model(state_tensor).argmax().item()
            
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
            
            state_next, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            replay_buffer.add(state, action, reward, state_next, done)
            state = state_next
            
            if frame_count % update_after_actions == 0 and len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states).max(1)[0]
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))
                loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            if frame_count % update_target_network == 0:
                target_model.load_state_dict(model.state_dict())
                
                log_file = f"{parent_dir}/logs/torch/modelstats_{datetime.now():%d-%m}.csv"
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
                        episode_count,
                        episode_reward,
                        running_reward,
                        np.max(episode_reward_history),
                        epsilon,
                        loss.item(),
                        (datetime.now().timestamp() - start_time)
                    ])
                
                torch.save(model.state_dict(), f"{parent_dir}/models/torch/space_invaders_qmodel_{episode_count}.pth")
            
            if done:
                break
        
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            episode_reward_history.pop(0)
        running_reward = np.mean(episode_reward_history)
        
        episode_count += 1
        
        if running_reward > 500:
            print(f"Solved at episode {episode_count}!")
            break
        
        if (max_episodes > 0 and episode_count >= max_episodes):
            print(f"Stopped at episode {episode_count}!")
            break
        
        if max_frames <= frame_count:
            print(f"Stopped at frame {frame_count}!")
            break

if __name__ == "__main__":
    train_dqn()
