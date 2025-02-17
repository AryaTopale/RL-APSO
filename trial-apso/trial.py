import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

class Drone:
    def __init__(self, search_space: Tuple[float, float], id: int):
        # Initialize drone at random position on the boundary
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            self.position = np.array([np.random.uniform(0, search_space[0]), search_space[1]])
        elif side == 1:  # Right
            self.position = np.array([search_space[0], np.random.uniform(0, search_space[1])])
        elif side == 2:  # Bottom
            self.position = np.array([np.random.uniform(0, search_space[0]), 0])
        else:  # Left
            self.position = np.array([0, np.random.uniform(0, search_space[1])])
            
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.best_position = self.position.copy()
        self.best_signal = float('-inf')
        self.id = id
        
    def update_best(self, signal_strength: float):
        if signal_strength > self.best_signal:
            self.best_signal = signal_strength
            self.best_position = self.position.copy()
            
class APSO:
    def __init__(self, 
                 n_drones: int = 5,
                 search_space: Tuple[float, float] = (100, 100),
                 w1: float = 0.675,
                 w2: float = -0.285,
                 c1: float = 1.193,
                 c2: float = 1.193,
                 T: float = 1.0,
                 source_power: float = 100,
                 alpha: float = 0.01):
        
        self.n_drones = n_drones
        self.search_space = search_space
        self.w1 = w1
        self.w2 = w2
        self.c1 = c1
        self.c2 = c2
        self.T = T
        self.source_power = source_power
        self.alpha = alpha
        
        # Initialize source at center
        self.source_position = np.array([search_space[0]/2, search_space[1]/2])
        
        # Initialize drones
        self.drones = [Drone(search_space, i) for i in range(n_drones)]
        self.global_best_position = None
        self.global_best_signal = float('-inf')
        
        # For tracking performance
        self.min_distances = []
        
    def measure_signal(self, position: np.ndarray) -> float:
        distance = np.linalg.norm(position - self.source_position)
        return self.source_power * np.exp(-self.alpha * distance**2)
    
    def update_drone(self, drone: Drone):
        # Update acceleration
        r1 = np.random.uniform(0, self.c1)
        r2 = np.random.uniform(0, self.c2)
        
        drone.acceleration = (self.w1 * drone.acceleration + 
                            r1 * (drone.best_position - drone.position) +
                            r2 * (self.global_best_position - drone.position))
        
        # Update velocity
        drone.velocity = self.w2 * drone.velocity + drone.acceleration * self.T
        
        # Update position
        drone.position = drone.position + drone.velocity * self.T
        
        # Ensure drone stays within bounds
        drone.position = np.clip(drone.position, [0, 0], [self.search_space[0], self.search_space[1]])
    
    def step(self) -> bool:
        # Update each drone's best position and global best
        for drone in self.drones:
            signal = self.measure_signal(drone.position)
            drone.update_best(signal)
            
            if signal > self.global_best_signal:
                self.global_best_signal = signal
                self.global_best_position = drone.position.copy()
        
        # Update drone positions
        for drone in self.drones:
            self.update_drone(drone)
            
        # Track minimum distance to source
        min_dist = min(np.linalg.norm(drone.position - self.source_position) 
                      for drone in self.drones)
        self.min_distances.append(min_dist)
        
        # Check if any drone has found the source
        return min_dist < 0.1
    
    def run(self, max_iterations: int = 1000) -> Tuple[int, float]:
        start_time = time.time()
        
        for iteration in range(max_iterations):
            if self.step():
                elapsed_time = time.time() - start_time
                return iteration + 1, elapsed_time
                
        elapsed_time = time.time() - start_time
        return max_iterations, elapsed_time
    
    def plot_performance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.min_distances)
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Distance to Source')
        plt.title('APSO Search Performance')
        plt.grid(True)
        plt.show()
        
    def visualize_search(self):
        plt.figure(figsize=(8, 8))
        
        # Plot search space boundaries
        plt.plot([0, self.search_space[0]], [0, 0], 'k-')
        plt.plot([0, self.search_space[0]], [self.search_space[1], self.search_space[1]], 'k-')
        plt.plot([0, 0], [0, self.search_space[1]], 'k-')
        plt.plot([self.search_space[0], self.search_space[0]], [0, self.search_space[1]], 'k-')
        
        # Plot source
        plt.plot(self.source_position[0], self.source_position[1], 'r*', markersize=15, label='Source')
        
        # Plot drones
        for drone in self.drones:
            plt.plot(drone.position[0], drone.position[1], 'bo', markersize=8)
            
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('APSO Search Visualization')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

# Run simulation
def main():
    # Create APSO instance
    apso = APSO(n_drones=5)
    
    # Run search
    iterations, elapsed_time = apso.run()
    
    print(f"Search completed in {iterations} iterations")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Plot results
    apso.plot_performance()
    apso.visualize_search()

if __name__ == "__main__":
    main()