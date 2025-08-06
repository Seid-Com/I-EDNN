import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import networkx as nx

class IsingModel:
    """
    Implementation of the 2D Ising model for physics simulations
    """
    
    def __init__(self, size=20, J=1.0, h=0.0, temperature=1.0, boundary='periodic'):
        """
        Initialize Ising model
        
        Args:
            size: Lattice size (size x size)
            J: Coupling strength (positive = ferromagnetic)
            h: External magnetic field
            temperature: Temperature parameter
            boundary: 'periodic' or 'open' boundary conditions
        """
        self.size = size
        self.J = J
        self.h = h
        self.temperature = temperature
        self.boundary = boundary
        self.beta = 1.0 / temperature if temperature > 0 else np.inf
        
        # Initialize random spin configuration
        self.lattice = 2 * np.random.randint(2, size=(size, size)) - 1
        
        # Pre-compute neighbor indices for efficiency
        self._compute_neighbors()
        
        # Energy and magnetization history
        self.energy_history = []
        self.magnetization_history = []
        
    def _compute_neighbors(self):
        """Pre-compute neighbor indices for each site"""
        self.neighbors = {}
        
        for i in range(self.size):
            for j in range(self.size):
                if self.boundary == 'periodic':
                    # Periodic boundary conditions
                    neighbors = [
                        ((i + 1) % self.size, j),      # Right
                        ((i - 1) % self.size, j),      # Left
                        (i, (j + 1) % self.size),      # Up
                        (i, (j - 1) % self.size)       # Down
                    ]
                else:
                    # Open boundary conditions
                    neighbors = []
                    if i < self.size - 1:
                        neighbors.append((i + 1, j))
                    if i > 0:
                        neighbors.append((i - 1, j))
                    if j < self.size - 1:
                        neighbors.append((i, j + 1))
                    if j > 0:
                        neighbors.append((i, j - 1))
                
                self.neighbors[(i, j)] = neighbors
    
    def local_energy(self, i, j):
        """
        Calculate local energy contribution of spin at (i, j)
        
        Args:
            i, j: Lattice coordinates
            
        Returns:
            Local energy
        """
        spin = self.lattice[i, j]
        neighbor_sum = sum(self.lattice[ni, nj] for ni, nj in self.neighbors[(i, j)])
        
        # Interaction energy with neighbors
        interaction_energy = -self.J * spin * neighbor_sum
        
        # External field energy
        field_energy = -self.h * spin
        
        return interaction_energy + field_energy
    
    def total_energy(self):
        """
        Calculate total energy of the system
        
        Returns:
            Total energy
        """
        energy = 0.0
        
        # Sum over all interactions (count each pair once)
        for i in range(self.size):
            for j in range(self.size):
                spin = self.lattice[i, j]
                
                # Count each neighbor pair once
                if self.boundary == 'periodic':
                    right_neighbor = self.lattice[(i + 1) % self.size, j]
                    down_neighbor = self.lattice[i, (j + 1) % self.size]
                    
                    energy -= self.J * spin * right_neighbor
                    energy -= self.J * spin * down_neighbor
                else:
                    if i < self.size - 1:
                        energy -= self.J * spin * self.lattice[i + 1, j]
                    if j < self.size - 1:
                        energy -= self.J * spin * self.lattice[i, j + 1]
                
                # External field
                energy -= self.h * spin
        
        return energy
    
    def magnetization(self):
        """
        Calculate total magnetization
        
        Returns:
            Total magnetization
        """
        return np.sum(self.lattice)
    
    def metropolis_step(self):
        """
        Perform one Monte Carlo step using Metropolis algorithm
        """
        # Choose random site
        i = np.random.randint(0, self.size)
        j = np.random.randint(0, self.size)
        
        # Calculate energy change for spin flip
        current_energy = self.local_energy(i, j)
        
        # Flip spin
        self.lattice[i, j] *= -1
        new_energy = self.local_energy(i, j)
        
        delta_E = new_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta_E > 0 and np.random.random() > np.exp(-self.beta * delta_E):
            # Reject the move
            self.lattice[i, j] *= -1
    
    def simulate(self, n_steps, record_interval=1):
        """
        Run Monte Carlo simulation
        
        Args:
            n_steps: Number of Monte Carlo steps
            record_interval: Interval for recording observables
        """
        for step in range(n_steps):
            self.metropolis_step()
            
            if step % record_interval == 0:
                self.energy_history.append(self.total_energy())
                self.magnetization_history.append(self.magnetization())
    
    def equilibrate(self, n_steps=1000):
        """
        Equilibrate the system (run without recording)
        
        Args:
            n_steps: Number of equilibration steps
        """
        for _ in range(n_steps):
            self.metropolis_step()
    
    def correlation_function(self, max_distance=None):
        """
        Calculate spin-spin correlation function
        
        Args:
            max_distance: Maximum distance to calculate
            
        Returns:
            distances, correlations
        """
        if max_distance is None:
            max_distance = self.size // 2
        
        correlations = []
        distances = []
        
        # Choose reference point at center
        ref_i, ref_j = self.size // 2, self.size // 2
        ref_spin = self.lattice[ref_i, ref_j]
        
        for r in range(1, max_distance + 1):
            corr_sum = 0
            count = 0
            
            for i in range(self.size):
                for j in range(self.size):
                    # Calculate distance
                    if self.boundary == 'periodic':
                        di = min(abs(i - ref_i), self.size - abs(i - ref_i))
                        dj = min(abs(j - ref_j), self.size - abs(j - ref_j))
                    else:
                        di = abs(i - ref_i)
                        dj = abs(j - ref_j)
                    
                    dist = np.sqrt(di**2 + dj**2)
                    
                    if abs(dist - r) < 0.5:  # Points at distance r
                        corr_sum += ref_spin * self.lattice[i, j]
                        count += 1
            
            if count > 0:
                correlations.append(corr_sum / count)
                distances.append(r)
        
        return np.array(distances), np.array(correlations)
    
    def susceptibility(self):
        """
        Calculate magnetic susceptibility
        
        Returns:
            Susceptibility
        """
        if len(self.magnetization_history) < 2:
            return 0.0
        
        mag_array = np.array(self.magnetization_history)
        mag_squared_mean = np.mean(mag_array**2)
        mag_mean_squared = np.mean(mag_array)**2
        
        return self.beta * (mag_squared_mean - mag_mean_squared) / (self.size**2)
    
    def specific_heat(self):
        """
        Calculate specific heat
        
        Returns:
            Specific heat
        """
        if len(self.energy_history) < 2:
            return 0.0
        
        energy_array = np.array(self.energy_history)
        energy_squared_mean = np.mean(energy_array**2)
        energy_mean_squared = np.mean(energy_array)**2
        
        return (self.beta**2) * (energy_squared_mean - energy_mean_squared) / (self.size**2)
    
    def visualize(self, title="Ising Configuration"):
        """
        Visualize the current spin configuration
        
        Args:
            title: Plot title
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.lattice, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title(f"{title} (T={self.temperature:.2f})")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def reset(self):
        """Reset the system to random configuration"""
        self.lattice = 2 * np.random.randint(2, size=(self.size, self.size)) - 1
        self.energy_history = []
        self.magnetization_history = []

class IsingGraph:
    """
    Ising model on arbitrary graphs (not just 2D lattices)
    """
    
    def __init__(self, graph, J=None, h=None, temperature=1.0):
        """
        Initialize Ising model on graph
        
        Args:
            graph: NetworkX graph or adjacency matrix
            J: Coupling matrix (or scalar for uniform coupling)
            h: External field vector (or scalar for uniform field)
            temperature: Temperature parameter
        """
        if isinstance(graph, np.ndarray):
            # Convert adjacency matrix to NetworkX graph
            self.graph = nx.from_numpy_array(graph)
        else:
            self.graph = graph
        
        self.n_nodes = self.graph.number_of_nodes()
        self.temperature = temperature
        self.beta = 1.0 / temperature if temperature > 0 else np.inf
        
        # Set couplings
        if J is None:
            self.J = np.ones((self.n_nodes, self.n_nodes))
        elif np.isscalar(J):
            self.J = J * nx.adjacency_matrix(self.graph).toarray()
        else:
            self.J = J
        
        # Set external fields
        if h is None:
            self.h = np.zeros(self.n_nodes)
        elif np.isscalar(h):
            self.h = h * np.ones(self.n_nodes)
        else:
            self.h = h
        
        # Initialize random spins
        self.spins = 2 * np.random.randint(2, size=self.n_nodes) - 1
        
        # History
        self.energy_history = []
        self.magnetization_history = []
    
    def local_energy(self, node):
        """
        Calculate local energy of a node
        
        Args:
            node: Node index
            
        Returns:
            Local energy
        """
        spin = self.spins[node]
        
        # Interaction energy
        interaction_energy = 0
        for neighbor in self.graph.neighbors(node):
            interaction_energy -= self.J[node, neighbor] * spin * self.spins[neighbor]
        
        # Field energy
        field_energy = -self.h[node] * spin
        
        return interaction_energy + field_energy
    
    def total_energy(self):
        """
        Calculate total energy of the system
        
        Returns:
            Total energy
        """
        energy = 0.0
        
        # Interaction energy (count each edge once)
        for edge in self.graph.edges():
            i, j = edge
            energy -= self.J[i, j] * self.spins[i] * self.spins[j]
        
        # Field energy
        energy -= np.sum(self.h * self.spins)
        
        return energy
    
    def metropolis_step(self):
        """
        Perform one Metropolis Monte Carlo step
        """
        # Choose random node
        node = np.random.randint(0, self.n_nodes)
        
        # Calculate energy change
        current_energy = self.local_energy(node)
        
        # Flip spin
        self.spins[node] *= -1
        new_energy = self.local_energy(node)
        
        delta_E = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_E > 0 and np.random.random() > np.exp(-self.beta * delta_E):
            # Reject move
            self.spins[node] *= -1
    
    def simulate(self, n_steps, record_interval=1):
        """
        Run simulation
        
        Args:
            n_steps: Number of steps
            record_interval: Recording interval
        """
        for step in range(n_steps):
            self.metropolis_step()
            
            if step % record_interval == 0:
                self.energy_history.append(self.total_energy())
                self.magnetization_history.append(np.sum(self.spins))

# Example usage and testing
if __name__ == "__main__":
    # Test 2D Ising model
    print("Testing 2D Ising Model...")
    
    # Low temperature (ordered phase)
    ising_low = IsingModel(size=20, J=1.0, temperature=1.0)
    ising_low.equilibrate(1000)
    ising_low.simulate(1000, record_interval=10)
    
    print(f"Low T: Final magnetization = {ising_low.magnetization()}")
    print(f"Low T: Final energy = {ising_low.total_energy()}")
    
    # High temperature (disordered phase)
    ising_high = IsingModel(size=20, J=1.0, temperature=5.0)
    ising_high.equilibrate(1000)
    ising_high.simulate(1000, record_interval=10)
    
    print(f"High T: Final magnetization = {ising_high.magnetization()}")
    print(f"High T: Final energy = {ising_high.total_energy()}")
    
    # Test graph Ising model
    print("\nTesting Graph Ising Model...")
    
    # Create random graph
    G = nx.erdos_renyi_graph(20, 0.3)
    ising_graph = IsingGraph(G, J=1.0, temperature=2.0)
    ising_graph.simulate(1000)
    
    print(f"Graph: Final magnetization = {np.sum(ising_graph.spins)}")
    print(f"Graph: Final energy = {ising_graph.total_energy()}")
