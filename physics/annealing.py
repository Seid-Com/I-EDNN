import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

class SimulatedAnnealing:
    """
    Simulated Annealing optimizer for Ising-like problems
    """
    
    def __init__(self, initial_temp=10.0, final_temp=0.01, cooling_schedule='exponential'):
        """
        Initialize simulated annealing
        
        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling_schedule: 'exponential', 'linear', or 'logarithmic'
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_schedule = cooling_schedule
        
        # History tracking
        self.temperature_history = []
        self.energy_history = []
        self.best_energy_history = []
        self.acceptance_rate_history = []
    
    def get_temperature(self, step, total_steps):
        """
        Get temperature at given step according to cooling schedule
        
        Args:
            step: Current step
            total_steps: Total number of steps
            
        Returns:
            Temperature at current step
        """
        progress = step / total_steps
        
        if self.cooling_schedule == 'exponential':
            # Exponential cooling: T(t) = T0 * (Tf/T0)^t
            alpha = (self.final_temp / self.initial_temp) ** (1.0 / total_steps)
            temp = self.initial_temp * (alpha ** step)
            
        elif self.cooling_schedule == 'linear':
            # Linear cooling: T(t) = T0 - (T0 - Tf) * t
            temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress
            
        elif self.cooling_schedule == 'logarithmic':
            # Logarithmic cooling: T(t) = T0 / log(1 + t)
            temp = self.initial_temp / np.log(1 + step + 1)
            temp = max(temp, self.final_temp)
            
        else:
            raise ValueError(f"Unknown cooling schedule: {self.cooling_schedule}")
        
        return max(temp, self.final_temp)
    
    def acceptance_probability(self, delta_E, temperature):
        """
        Calculate acceptance probability for energy change
        
        Args:
            delta_E: Energy change
            temperature: Current temperature
            
        Returns:
            Acceptance probability
        """
        if delta_E <= 0:
            return 1.0
        elif temperature <= 0:
            return 0.0
        else:
            return np.exp(-delta_E / temperature)
    
    def anneal_step(self, spins, J_matrix, temperature, h=None):
        """
        Perform one annealing step on spin configuration
        
        Args:
            spins: Current spin configuration
            J_matrix: Interaction matrix
            temperature: Current temperature
            h: External field (optional)
            
        Returns:
            Updated spin configuration
        """
        if h is None:
            h = np.zeros(len(spins))
        
        new_spins = spins.copy()
        n_spins = len(spins)
        n_accepted = 0
        
        # Try to flip each spin
        for _ in range(n_spins):
            # Choose random spin
            i = np.random.randint(0, n_spins)
            
            # Calculate current energy contribution
            current_energy = -0.5 * np.sum(J_matrix[i, :] * spins[i] * spins) - h[i] * spins[i]
            
            # Flip spin and calculate new energy
            new_spins[i] = -spins[i]
            new_energy = -0.5 * np.sum(J_matrix[i, :] * new_spins[i] * new_spins) - h[i] * new_spins[i]
            
            delta_E = new_energy - current_energy
            
            # Accept or reject move
            if np.random.random() < self.acceptance_probability(delta_E, temperature):
                spins[i] = new_spins[i]
                n_accepted += 1
            else:
                new_spins[i] = spins[i]  # Revert
        
        acceptance_rate = n_accepted / n_spins
        return spins, acceptance_rate
    
    def optimize_ising(self, J_matrix, initial_spins=None, h=None, n_steps=1000):
        """
        Optimize Ising model using simulated annealing
        
        Args:
            J_matrix: Interaction matrix
            initial_spins: Initial spin configuration
            h: External field
            n_steps: Number of annealing steps
            
        Returns:
            Optimized spin configuration, final energy, optimization history
        """
        n_spins = J_matrix.shape[0]
        
        if initial_spins is None:
            spins = 2 * np.random.randint(2, size=n_spins) - 1
        else:
            spins = initial_spins.copy()
        
        if h is None:
            h = np.zeros(n_spins)
        
        # Track best solution
        best_spins = spins.copy()
        best_energy = self._calculate_energy(spins, J_matrix, h)
        
        # Clear history
        self.temperature_history = []
        self.energy_history = []
        self.best_energy_history = []
        self.acceptance_rate_history = []
        
        for step in range(n_steps):
            # Get current temperature
            temperature = self.get_temperature(step, n_steps)
            
            # Perform annealing step
            spins, acceptance_rate = self.anneal_step(spins, J_matrix, temperature, h)
            
            # Calculate energy
            energy = self._calculate_energy(spins, J_matrix, h)
            
            # Update best solution
            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()
            
            # Record history
            self.temperature_history.append(temperature)
            self.energy_history.append(energy)
            self.best_energy_history.append(best_energy)
            self.acceptance_rate_history.append(acceptance_rate)
        
        return best_spins, best_energy, {
            'temperature': self.temperature_history,
            'energy': self.energy_history,
            'best_energy': self.best_energy_history,
            'acceptance_rate': self.acceptance_rate_history
        }
    
    def _calculate_energy(self, spins, J_matrix, h):
        """
        Calculate total energy of spin configuration
        
        Args:
            spins: Spin configuration
            J_matrix: Interaction matrix
            h: External field
            
        Returns:
            Total energy
        """
        interaction_energy = -0.5 * np.sum(J_matrix * np.outer(spins, spins))
        field_energy = -np.sum(h * spins)
        return interaction_energy + field_energy

class AdaptiveAnnealing(SimulatedAnnealing):
    """
    Adaptive simulated annealing with dynamic temperature adjustment
    """
    
    def __init__(self, initial_temp=10.0, final_temp=0.01, 
                 target_acceptance_rate=0.5, adaptation_rate=0.1):
        """
        Initialize adaptive annealing
        
        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            target_acceptance_rate: Desired acceptance rate
            adaptation_rate: Rate of temperature adaptation
        """
        super().__init__(initial_temp, final_temp, 'exponential')
        self.target_acceptance_rate = target_acceptance_rate
        self.adaptation_rate = adaptation_rate
        self.current_temp = initial_temp
    
    def adapt_temperature(self, acceptance_rate):
        """
        Adapt temperature based on acceptance rate
        
        Args:
            acceptance_rate: Current acceptance rate
        """
        if acceptance_rate > self.target_acceptance_rate:
            # Too many acceptances, decrease temperature
            self.current_temp *= (1 - self.adaptation_rate)
        else:
            # Too few acceptances, increase temperature
            self.current_temp *= (1 + self.adaptation_rate)
        
        # Ensure temperature stays within bounds
        self.current_temp = max(self.current_temp, self.final_temp)
        self.current_temp = min(self.current_temp, self.initial_temp)
    
    def optimize_ising(self, J_matrix, initial_spins=None, h=None, n_steps=1000):
        """
        Optimize with adaptive temperature
        """
        n_spins = J_matrix.shape[0]
        
        if initial_spins is None:
            spins = 2 * np.random.randint(2, size=n_spins) - 1
        else:
            spins = initial_spins.copy()
        
        if h is None:
            h = np.zeros(n_spins)
        
        # Reset current temperature
        self.current_temp = self.initial_temp
        
        # Track best solution
        best_spins = spins.copy()
        best_energy = self._calculate_energy(spins, J_matrix, h)
        
        # Clear history
        self.temperature_history = []
        self.energy_history = []
        self.best_energy_history = []
        self.acceptance_rate_history = []
        
        for step in range(n_steps):
            # Perform annealing step with current temperature
            spins, acceptance_rate = self.anneal_step(spins, J_matrix, self.current_temp, h)
            
            # Adapt temperature
            if step > 100:  # Allow some initial exploration
                self.adapt_temperature(acceptance_rate)
            
            # Calculate energy
            energy = self._calculate_energy(spins, J_matrix, h)
            
            # Update best solution
            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()
            
            # Record history
            self.temperature_history.append(self.current_temp)
            self.energy_history.append(energy)
            self.best_energy_history.append(best_energy)
            self.acceptance_rate_history.append(acceptance_rate)
        
        return best_spins, best_energy, {
            'temperature': self.temperature_history,
            'energy': self.energy_history,
            'best_energy': self.best_energy_history,
            'acceptance_rate': self.acceptance_rate_history
        }

class QuantumAnnealing:
    """
    Simplified quantum annealing simulation
    """
    
    def __init__(self, transverse_field_schedule=None):
        """
        Initialize quantum annealing
        
        Args:
            transverse_field_schedule: Function that returns transverse field strength
                                     as function of progress (0 to 1)
        """
        if transverse_field_schedule is None:
            # Default: linear decrease from 1 to 0
            self.transverse_field_schedule = lambda s: 1.0 - s
        else:
            self.transverse_field_schedule = transverse_field_schedule
    
    def quantum_fluctuations(self, spins, transverse_field):
        """
        Apply quantum fluctuations (simplified)
        
        Args:
            spins: Current spin configuration
            transverse_field: Strength of transverse field
            
        Returns:
            Modified spin configuration
        """
        new_spins = spins.copy()
        n_spins = len(spins)
        
        # Probability of quantum tunneling
        flip_probability = transverse_field * 0.1  # Simplified model
        
        for i in range(n_spins):
            if np.random.random() < flip_probability:
                new_spins[i] *= -1
        
        return new_spins
    
    def optimize_ising(self, J_matrix, initial_spins=None, h=None, n_steps=1000):
        """
        Optimize using quantum annealing
        
        Args:
            J_matrix: Interaction matrix
            initial_spins: Initial spin configuration
            h: External field
            n_steps: Number of annealing steps
            
        Returns:
            Optimized spin configuration and history
        """
        n_spins = J_matrix.shape[0]
        
        if initial_spins is None:
            spins = 2 * np.random.randint(2, size=n_spins) - 1
        else:
            spins = initial_spins.copy()
        
        if h is None:
            h = np.zeros(n_spins)
        
        best_spins = spins.copy()
        best_energy = self._calculate_energy(spins, J_matrix, h)
        
        energy_history = []
        transverse_field_history = []
        
        for step in range(n_steps):
            progress = step / n_steps
            transverse_field = self.transverse_field_schedule(progress)
            
            # Apply quantum fluctuations
            candidate_spins = self.quantum_fluctuations(spins, transverse_field)
            
            # Calculate energies
            current_energy = self._calculate_energy(spins, J_matrix, h)
            candidate_energy = self._calculate_energy(candidate_spins, J_matrix, h)
            
            # Accept based on energy improvement and quantum effects
            accept_prob = 1.0 if candidate_energy <= current_energy else transverse_field
            
            if np.random.random() < accept_prob:
                spins = candidate_spins
                current_energy = candidate_energy
            
            # Update best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_spins = spins.copy()
            
            energy_history.append(current_energy)
            transverse_field_history.append(transverse_field)
        
        return best_spins, best_energy, {
            'energy': energy_history,
            'transverse_field': transverse_field_history
        }
    
    def _calculate_energy(self, spins, J_matrix, h):
        """Calculate energy (same as classical)"""
        interaction_energy = -0.5 * np.sum(J_matrix * np.outer(spins, spins))
        field_energy = -np.sum(h * spins)
        return interaction_energy + field_energy

def compare_annealing_methods(J_matrix, h=None, n_steps=1000, n_runs=5):
    """
    Compare different annealing methods
    
    Args:
        J_matrix: Interaction matrix
        h: External field
        n_steps: Number of annealing steps
        n_runs: Number of runs for statistics
        
    Returns:
        Comparison results
    """
    methods = {
        'Classical SA': SimulatedAnnealing(),
        'Adaptive SA': AdaptiveAnnealing(),
        'Quantum Annealing': QuantumAnnealing()
    }
    
    results = {}
    
    for method_name, method in methods.items():
        print(f"Testing {method_name}...")
        
        energies = []
        times = []
        
        for run in range(n_runs):
            start_time = time.time()
            
            if method_name == 'Quantum Annealing':
                best_spins, best_energy, history = method.optimize_ising(
                    J_matrix, h=h, n_steps=n_steps
                )
            else:
                best_spins, best_energy, history = method.optimize_ising(
                    J_matrix, h=h, n_steps=n_steps
                )
            
            end_time = time.time()
            
            energies.append(best_energy)
            times.append(end_time - start_time)
        
        results[method_name] = {
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'best_energy': np.min(energies),
            'mean_time': np.mean(times),
            'energies': energies
        }
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test simulated annealing
    print("Testing Simulated Annealing...")
    
    # Create random Ising problem
    n_spins = 20
    J_matrix = np.random.randn(n_spins, n_spins) * 0.1
    J_matrix = (J_matrix + J_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(J_matrix, 0)  # Zero diagonal
    
    # Test classical annealing
    sa = SimulatedAnnealing(initial_temp=5.0, final_temp=0.01)
    best_spins, best_energy, history = sa.optimize_ising(J_matrix, n_steps=1000)
    
    print(f"Classical SA: Best energy = {best_energy:.4f}")
    
    # Test adaptive annealing
    adaptive_sa = AdaptiveAnnealing(initial_temp=5.0, final_temp=0.01)
    best_spins_adaptive, best_energy_adaptive, history_adaptive = adaptive_sa.optimize_ising(
        J_matrix, n_steps=1000
    )
    
    print(f"Adaptive SA: Best energy = {best_energy_adaptive:.4f}")
    
    # Test quantum annealing
    qa = QuantumAnnealing()
    best_spins_quantum, best_energy_quantum, history_quantum = qa.optimize_ising(
        J_matrix, n_steps=1000
    )
    
    print(f"Quantum Annealing: Best energy = {best_energy_quantum:.4f}")
    
    # Compare methods
    print("\nComparing methods...")
    comparison = compare_annealing_methods(J_matrix, n_steps=500, n_runs=3)
    
    for method, results in comparison.items():
        print(f"{method}: Mean energy = {results['mean_energy']:.4f} ± {results['std_energy']:.4f}")
