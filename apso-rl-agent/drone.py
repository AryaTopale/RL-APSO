import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class Drone:
    def __init__(self, search_space, id):
        side = np.random.randint(0, 4)
        if side == 0:
            self.position = np.array([np.random.uniform(0, search_space[0]), search_space[1]])
        elif side == 1:
            self.position = np.array([search_space[0], np.random.uniform(0, search_space[1])])
        elif side == 2:
            self.position = np.array([np.random.uniform(0, search_space[0]), 0])
        else:
            self.position = np.array([0, np.random.uniform(0, search_space[1])])

        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.best_position = self.position.copy()
        self.best_signal = float('-inf')
        self.id = id

    def update_best(self, signal_strength):
        if signal_strength > self.best_signal:
            self.best_signal = signal_strength
            self.best_position = self.position.copy()

class APSO:
    def __init__(self, n_drones=5, search_space=(100, 100), w1=0.675, w2=-0.285, c1=1.193, c2=1.193, T=1.0, source_power=100, alpha=0.01):
        self.n_drones = n_drones
        self.search_space = search_space
        self.w1, self.w2, self.c1, self.c2 = w1, w2, c1, c2
        self.T = T
        self.source_power = source_power
        self.alpha = alpha
        self.source_position = np.array([search_space[0] / 2, search_space[1] / 2])
        self.drones = [Drone(search_space, i) for i in range(n_drones)]
        self.global_best_position = None
        self.global_best_signal = float('-inf')
        self.min_distances = []
        self.prev_min_distance = float('inf')
        self.steps_without_improvement = 0

    def measure_signal(self, position):
        distance = np.linalg.norm(position - self.source_position)
        return self.source_power * np.exp(-self.alpha * distance**2)

    def update_drone(self, drone):
        r1, r2 = np.random.uniform(0, self.c1), np.random.uniform(0, self.c2)
        drone.acceleration = (self.w1 * drone.acceleration +
                            r1 * (drone.best_position - drone.position) +
                            r2 * (self.global_best_position - drone.position))
        drone.velocity = self.w2 * drone.velocity + drone.acceleration * self.T
        drone.position = np.clip(drone.position + drone.velocity * self.T, [0, 0], [self.search_space[0], self.search_space[1]])

    def get_swarm_metrics(self):
        positions = np.array([drone.position for drone in self.drones])
        centroid = np.mean(positions, axis=0)
        distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        swarm_radius = np.max(distances_to_centroid)
        swarm_density = np.mean(distances_to_centroid)
        return swarm_radius, swarm_density

    def step(self):
        for drone in self.drones:
            signal = self.measure_signal(drone.position)
            drone.update_best(signal)
            if signal > self.global_best_signal:
                self.global_best_signal = signal
                self.global_best_position = drone.position.copy()

        for drone in self.drones:
            self.update_drone(drone)

        min_dist = min(np.linalg.norm(drone.position - self.source_position) for drone in self.drones)
        
        # Track improvement
        if min_dist < self.prev_min_distance:
            self.steps_without_improvement = 0
            self.prev_min_distance = min_dist
        else:
            self.steps_without_improvement += 1
            
        self.min_distances.append(min_dist)
        return min_dist < 0.1

class APSOEnv(gym.Env):
    def __init__(self):
        super(APSOEnv, self).__init__()
        self.action_space = spaces.Box(low=-0.5, high=2, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.apso = None
        self.episode_steps = 0
        self.max_steps = 200  # Reduced episode length
        self.reset()

    def reset(self):
        self.apso = APSO()
        self.episode_steps = 0
        return self.get_observation()

    def get_observation(self):
        swarm_radius, swarm_density = self.apso.get_swarm_metrics()
        return np.array([
            self.apso.w1, 
            self.apso.w2, 
            self.apso.c1, 
            self.apso.c2,
            np.mean([np.linalg.norm(d.position - self.apso.source_position) for d in self.apso.drones]),
            np.std([np.linalg.norm(d.position - self.apso.source_position) for d in self.apso.drones]),
            swarm_radius,
            swarm_density
        ])

    def calculate_reward(self, done):
        current_min_dist = min(np.linalg.norm(d.position - self.apso.source_position) 
                             for d in self.apso.drones)
        
        # Base reward based on distance improvement
        if self.apso.steps_without_improvement == 0:
            improvement_reward = 10.0 * (self.apso.prev_min_distance - current_min_dist)
        else:
            improvement_reward = 0
        
        # Distance-based reward component
        distance_reward = -0.1 * current_min_dist
        
        # Swarm behavior rewards
        swarm_radius, swarm_density = self.apso.get_swarm_metrics()
        exploration_reward = 0.1 * swarm_radius if current_min_dist > 20 else 0
        
        # Early convergence penalty
        early_convergence_penalty = -5.0 if swarm_radius < 5 and current_min_dist > 10 else 0
        
        # Stability reward
        stability_reward = 2.0 if self.jury_stability_test(self.apso.w1, self.apso.w2, 
                                                         self.apso.c1, self.apso.c2) else -2.0
        
        # Combine all rewards
        reward = (
            distance_reward +
            improvement_reward +
            exploration_reward +
            early_convergence_penalty +
            stability_reward
        )
        
        # Success bonus
        if done:
            reward += 2000
            
        return reward

    def step(self, action):
        self.episode_steps += 1
        self.apso.w1, self.apso.w2, self.apso.c1, self.apso.c2 = np.clip(action, -0.5, 2)
        done = self.apso.step()
        
        # Additional termination conditions
        current_min_dist = min(np.linalg.norm(d.position - self.apso.source_position) 
                             for d in self.apso.drones)
        
        # End episode if:
        # 1. Found the source
        # 2. Max steps reached
        # 3. Stuck in local minimum (too many steps without improvement and far from source)
        done = done or \
               self.episode_steps >= self.max_steps or \
               (self.apso.steps_without_improvement > 50 and current_min_dist > 20)
        
        reward = self.calculate_reward(done)
        return self.get_observation(), reward, done, {}

    def jury_stability_test(self, w1, w2, c1, c2):
        return (abs(w1) < 1) and (abs(w2) < 1) and (c1 > 0) and (c2 > 0)

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self):
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True

def train_rl():
    env = APSOEnv()
    
    # Create callback
    reward_callback = RewardCallback()
    
    # Modified PPO parameters
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
    )
    
    model.learn(total_timesteps=1000000, callback=reward_callback)
    
    # Plot rewards
    episodes = np.arange(len(reward_callback.episode_rewards))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, reward_callback.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Episode Reward')
    plt.title('Mean Episode Reward vs Episode')
    plt.grid(True)
    plt.savefig('reward_plot.png')
    plt.close()
    
    return model, reward_callback.episode_rewards

def evaluate_rl(model, n_evaluations=5):
    all_distances = []
    
    for eval_num in range(n_evaluations):
        env = APSOEnv()
        obs = env.reset()
        distances = []
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            min_dist = min(np.linalg.norm(drone.position - env.apso.source_position) 
                          for drone in env.apso.drones)
            distances.append(min_dist)
            
            if done:
                print(f"Evaluation {eval_num + 1} completed in {step + 1} steps")
                break
        
        all_distances.append(distances)
    
    plt.figure(figsize=(12, 8))
    
    for i, distances in enumerate(all_distances):
        plt.plot(distances, alpha=0.3, label=f'Run {i+1}')
    
    mean_distances = np.mean([d[:min(map(len, all_distances))] for d in all_distances], axis=0)
    plt.plot(mean_distances, 'r-', linewidth=2, label='Mean Performance')
    
    plt.xlabel("Steps")
    plt.ylabel("Min Distance to Source")
    plt.title("RL Optimized APSO Performance")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.savefig('performance_plot.png')
    plt.close()
    
    final_distances = [d[-1] for d in all_distances]
    print(f"\nFinal Statistics:")
    print(f"Mean final distance: {np.mean(final_distances):.2f}")
    print(f"Best final distance: {np.min(final_distances):.2f}")
    print(f"Worst final distance: {np.max(final_distances):.2f}")

if __name__ == "__main__":
    model, episode_rewards = train_rl()
    evaluate_rl(model)