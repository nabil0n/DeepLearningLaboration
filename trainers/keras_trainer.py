# %% [markdown]
# ### Based on Lec5-RL-Gymnasium.py

# %%
import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
import tensorflow as tf
import ale_py
from collections import deque
import csv
import os
from datetime import datetime

gym.register_envs(ale_py)

# %%
seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = (epsilon_max - epsilon_min)
batch_size = 32
max_steps_per_episode = 10000
max_episodes = 0
max_frames = 1e7

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")

env = AtariPreprocessing(env)

env = FrameStack(env, 4)
def trigger(t):
    return t % 100 == 0
env = gym.wrappers.RecordVideo(env, video_folder="../videos/keras", episode_trigger=trigger, disable_logger=True)


# %%
def preprocess_observation(observation):
    observation = tf.transpose(observation, perm=[0, 1, 2])
    observation = tf.cast(observation, tf.float32) / 255.0
    return observation

# %%
height, width, channels = env.observation_space.shape
env.observation_space.shape

# %%
num_actions = 6

def create_q_model():
    return keras.Sequential(
        [
            # layers.InputLayer(shape=(3, height, width, channels)),
            layers.Permute((2, 3, 1)),  # Rearrange dimensions
            layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"),
            layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(num_actions, activation="linear")
        ]
    )


model = create_q_model()
model_target = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# observation, _ = env.reset(seed=42)
# state = np.array(observation)
# state_tensor = keras.ops.convert_to_tensor(state)
# state_tensor = keras.ops.expand_dims(state_tensor, 0)
# print(state_tensor)

# %%
model.summary()

# %%
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

max_buffer_size = 100000
action_history = deque(maxlen=max_buffer_size)
state_history = deque(maxlen=max_buffer_size)
state_next_history = deque(maxlen=max_buffer_size)
done_history = deque(maxlen=max_buffer_size)
rewards_history = deque(maxlen=max_buffer_size)

# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 1000000
# Train the model after 4 actions
update_after_actions = 6
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

# %%
start_time = datetime.now().timestamp()

while True:
    observation, _ = env.reset()
    state = np.array(observation)
    episode_reward = 0
    # print(observation.shape)

    # break
    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = keras.ops.convert_to_tensor(state)
            state_tensor = keras.ops.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = keras.ops.argmax(action_probs[0]).numpy()
        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _, _ = env.step(action)

        episode_reward += reward

        # Save actions and states in replay buffer
        # state_processed = preprocess_observation(state)
        # state_next_processed = preprocess_observation(state_next)
        state_processed = state
        state_next_processed = state_next
        # print(state_tensor.shape)

        # Append data to replay buffer
        action_history.append(action)
        state_history.append(state_processed)
        state_next_history.append(state_next_processed)
        done_history.append(float(done))
        rewards_history.append(float(reward))
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(
                range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([np.array(state_history[i]) for i in indices])
            state_next_sample = np.array(
                [np.array(state_next_history[i]) for i in indices])
            rewards_sample = np.array([rewards_history[i] for i in indices], dtype=np.float32)
            action_sample = np.array([action_history[i] for i in indices], dtype=np.int32)
            done_sample = keras.ops.convert_to_tensor(
                [float(done_history[i]) for i in indices], dtype=tf.float32
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample, verbose=0)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * keras.ops.amax(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * \
                (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = keras.ops.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = keras.ops.sum(
                    keras.ops.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            log_file = f"../logs/modelstats_{datetime.now():%d-%m}.csv"
            print(f"best score of last 100: {np.max(episode_reward_history)}, running reward: {running_reward:.2f} at episode {episode_count}, frame count {frame_count}, time: {datetime.now().timestamp()-start_time}")
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if os.stat(log_file).st_size == 0:
                    writer.writerow(["episode", "frame", "running_reward", "max_reward", "time"])
                writer.writerow([episode_count, frame_count, running_reward, np.max(episode_reward_history), (datetime.now().timestamp()-start_time)])
                f.flush()
            model.save(f"../models/space_invaders_qmodel_{episode_count}.keras")

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 500:
        print("Solved at episode {}!".format(episode_count))
        break

    if (
        max_episodes > 0 and episode_count >= max_episodes
    ):  # Maximum number of episodes reached
        print("Stopped at episode {}!".format(episode_count))
        break
    if (max_frames <= frame_count):
        print(f"Stopped at frame {frame_count}!")


