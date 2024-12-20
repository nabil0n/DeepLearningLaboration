import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import ale_py
from collections import deque
import csv
import os
from datetime import datetime
import pickle

gym.register_envs(ale_py)

batch_size = 32
max_steps_per_episode = 10000
max_episodes = 0
max_frames = 1e7
num_frames = 4

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

env = gym.wrappers.RecordVideo(
    env,
    video_folder=f"{parent_dir}/videos/jax",
    episode_trigger=lambda episode: episode % 100 == 0,
    disable_logger=True
)

class DQN(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_actions)(x)
        return x

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, key):
        batch_indices = jax.random.choice(key, len(self.buffer), shape=(batch_size,), replace=False)
        batch = [self.buffer[i] for i in batch_indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            jnp.array(states),
            jnp.array(actions),
            jnp.array(rewards, dtype=jnp.float32),
            jnp.array(next_states),
            jnp.array(dones, dtype=jnp.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

def compute_loss(params, state, action, reward, next_state, done, model, target_model, gamma):
    q_values = model.apply({'params': params}, state)
    current_q_values = q_values[jnp.arange(q_values.shape[0]), action]
    
    next_q_values = target_model.apply({'params': params}, next_state)
    next_max_q_values = jnp.max(next_q_values, axis=1)
    
    target_q_values = reward + gamma * next_max_q_values * (1 - done)
    
    loss = optax.losses.huber_loss(current_q_values, target_q_values).mean()
    return loss

compute_loss_grad = jax.value_and_grad(compute_loss)

def train_dqn():
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_interval = (epsilon_max - epsilon_min)
    num_actions = 6

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    model = DQN(num_actions=num_actions)
    target_model = DQN(num_actions=num_actions)
    
    sample_input = jnp.zeros((num_frames, 84, 84))
    variables = model.init(subkey, sample_input)
    
    optimizer = optax.adam(learning_rate=0.00025)
    
    state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=variables['params'], 
        tx=optimizer
    )
    
    target_state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=variables['params'], 
        tx=optimizer
    )
    
    replay_buffer = ReplayBuffer(max_size=100000)
    
    epsilon_random_frames = 50000
    epsilon_greedy_frames = 1000000.0
    update_after_actions = 6
    update_target_network = 10000
    
    frame_count = 0
    episode_count = 0
    episode_reward_history = []
    running_reward = 0
    
    start_time = datetime.now().timestamp()
    
    while True:
        observation, _ = env.reset()
        state_array = np.array(observation)
        episode_reward = 0
        
        for timestep in range(1, max_steps_per_episode):
            frame_count += 1
            
            key, subkey = jax.random.split(key)
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                action = np.random.choice(num_actions)
            else:
                state_tensor = jnp.array(state_array)[None, ...]
                action = jnp.argmax(model.apply({'params': state.params}, state_tensor)[0]).item()
            
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
            
            state_next, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            replay_buffer.add(state_array, action, reward, state_next, done)
            state_array = state_next
            
            if frame_count % update_after_actions == 0 and len(replay_buffer) > batch_size:
                key, subkey = jax.random.split(key)
                
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                    replay_buffer.sample(batch_size, subkey)
                
                loss, grads = compute_loss_grad(
                    state.params, 
                    batch_states, 
                    batch_actions, 
                    batch_rewards, 
                    batch_next_states, 
                    batch_dones, 
                    model, 
                    model,  # HUMAN ERROR: Should be target_model
                    gamma
                )
                
                state = state.apply_gradients(grads=grads)
            
            if frame_count % update_target_network == 0:
                target_state = target_state.replace(params=state.params)
                
                log_file = f"{parent_dir}/logs/jax/modelstats_{datetime.now():%d-%m}.csv"
                
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
                        np.max(episode_reward_history) if episode_reward_history else 0,
                        epsilon,
                        loss.item(),
                        (datetime.now().timestamp() - start_time)
                    ])
                
                with open(f"{parent_dir}/models/jax/space_invaders_qmodel_{episode_count}.pkl", 'wb') as f:
                    pickle.dump(state.params, f)
            
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
        
        if (max_episodes > 0 and episode_count >= max_episodes) or (max_frames <= frame_count):
            print(f"Stopped at {'episode' if max_episodes > 0 else 'frame'} {episode_count or frame_count}!")
            break

if __name__ == "__main__":
    train_dqn()
