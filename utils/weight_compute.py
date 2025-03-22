from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm

from .data_loader import load_raw_data
from .data_process import process_dataset
from .hawkes import HawkesModelHandler


class WeightCalculator:
    """Core class for weight calculation using Hawkes process intensities"""
    
    def __init__(self, params: Dict):
        """
        Args:
            params: Configuration dictionary containing:
                - a: Threshold parameter for phi_derivative
                - b: Cutoff parameter for phi_derivative
                - rho1: Scaling factor for negative inputs
                - rho2: Scaling factor for positive inputs
        """
        self.a = params['a']
        self.b = params['b']
        self.rho1 = params['rho1']
        self.rho2 = params['rho2']
        
        self._validate_params()
        
    def _validate_params(self):
        """Validate parameter constraints"""
        if any(val <= 0 for val in [self.a, self.b, self.rho1, self.rho2]):
            raise ValueError("All coefficients must be positive")

    def phi_derivative(self, x: float) -> float:
        """Calculate φ'(x) using piecewise definition
        
        Args:
            x: Input value
            a: First segment threshold
            b: Second segment cutoff
            
        Returns:
            Derivative value at x
        """
        if 0 <= x <= self.a:
            return (1 + x) / (1 + x + x**2 / 2)
        elif self.a < x <= self.b:
            base_value = self.phi_derivative(self.a)
            return base_value * ((self.b - x) ** 2) / ((self.b - self.a) ** 2)
        else:
            return 0.0

    def phi_prime(self, x: float) -> float:
        """Scaled version of φ' function with range handling
        
        Args:
            x: Input value (must be >= -1)
            
        Returns:
            Scaled derivative value
            
        Raises:
            ValueError: For inputs < -1
        """
        if x < -1:
            raise ValueError(f"Input x({x}) must be >= -1")
            
        return self.phi_derivative(x/self.rho2) if x >= 0 else self._solve_negative_phi_prime(x)

    def _solve_negative_phi_prime(self, x: float) -> float:
        """Solve equation for x < 0 cases using numerical optimization"""
        def equation(x_prime: float) -> float:
            left = (x_prime + 1) * np.exp(-x_prime - 1)
            right = (x + 1) * np.exp(-x - 1)
            return left - right

        x_prime_guess = max(x, -0.9)  # Avoid numerical issues near -1
        result = fsolve(equation, x0=x_prime_guess, full_output=True)
        return self.phi_derivative(result[0][0] / self.rho1)

    @staticmethod
    def lambda_k(
        u: float,
        baseline: float,
        adjacency_vector: np.ndarray,
        decay_vector: np.ndarray,
        event_times: List[np.ndarray]
    ) -> float:
        """Calculate Hawkes process intensity at time u
        
        Args:
            u: Current time point
            baseline: Base intensity
            adjacency_vector: Influence coefficients
            decay_vector: Decay rates
            event_times: Historical event timestamps
            
        Returns:
            Total intensity value at u
        """
        intensity = baseline
        for j, events in enumerate(event_times):
            intensity += sum(
                adjacency_vector[j] * np.exp(-decay_vector[j] * (u - t_j))
                for t_j in events if t_j <= u
            )
        return intensity

    def compute_w(
        self,
        baseline: np.ndarray,
        t_points: np.ndarray,
        adjacency_matrix: np.ndarray,
        decay_matrix: np.ndarray,
        event_times: List[List[np.ndarray]]
    ) -> np.ndarray:
        """Compute weight sequence for time windows
        
        Args:
            baseline: Base intensity vector
            t_points: Time point sequence
            adjacency_matrix: Influence coefficient matrix
            decay_matrix: Decay rate matrix
            event_times: Event timestamp lists
            
        Returns:
            Weight value array
        """
        def calculate_w_i(i: int) -> float:
            t_start, t_end = t_points[i-1], t_points[i]
            
            total_lambda = lambda u: sum(
                self.lambda_k(u, baseline[k], adjacency_matrix[k], 
                            decay_matrix[k], event_times)
                for k in range(len(baseline))
            )
            
            integral_value, _ = quad(total_lambda, t_start, t_end)
            return self.phi_prime(integral_value - 1)

        return np.array(Parallel(n_jobs=-1)(
            delayed(calculate_w_i)(i) for i in range(1, len(t_points))
        ))

class WeightAnalyzer:
    """Utility class for analyzing weight distributions"""
    
    @staticmethod
    def compute_zero_ratio(weights: List[np.ndarray]) -> Tuple[float, List[float]]:
        """Calculate zero-weight proportions
        
        Args:
            weights: List of weight arrays (per sample)
            
        Returns:
            Tuple: (Overall zero ratio, individual dataset ratios)
        """
        merged = np.concatenate(weights)
        total_ratio = np.mean(merged == 0)
        return total_ratio, [np.mean(arr == 0) for arr in weights]

    @staticmethod
    def format_result(total_ratio: float, individual_ratios: List[float]) -> str:
        """Format analysis results for reporting"""
        return (
            f"Zero-weight Ratios - Train: {individual_ratios[0]:.2%}, "
            f"Valid: {individual_ratios[1]:.2%}, "
            f"Test: {individual_ratios[2]:.2%}\n"
            f"Overall Ratio: {total_ratio:.2%}"
        )

if __name__ == "__main__":
    # Example configuration
    CONFIG = {
        'a': 0.5,
        'b': 2.0,
        'rho1': 1.0,
        'rho2': 1.0
    }
    
    # Initialize calculator
    calculator = WeightCalculator(CONFIG)
    
    MAX_ITER = 500

    # Load and process data
    raw_data = load_raw_data("example/configs/origin_earthquake.yaml", 'earthquake')
    processed_data = process_dataset("example/configs/origin_earthquake.yaml", "earthquake")
    num_event_types = raw_data["train"]["dim_process"]
    
    # Model initialization and training
    handler = HawkesModelHandler(max_iter=MAX_ITER, random_seed=42)
    decay_matrix = handler.create_decay_matrix(num_event_types, 4, 0.01)

    model = handler.initialize_model()
    trained_model = handler.train_model(
        model, 
        processed_data['train']['buckets']
    )
    
    # Weight computation
    dataset_weights = {
        'train': [
            calculator.compute_w(
                baseline=trained_model.baseline,
                t_points=seq,
                adjacency_matrix=trained_model.adjacency,
                decay_matrix=trained_model.decays,
                event_times=processed_data['train']['buckets'][i]
            )
            for i, seq in tqdm(enumerate(processed_data['train']['sequences']['time_seqs']),
                desc="Processing train set",
                total=len(processed_data['train']['sequences']['time_seqs']),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        ],
        'valid': [
            calculator.compute_w(
                baseline=trained_model.baseline,
                t_points=seq,
                adjacency_matrix=trained_model.adjacency,
                decay_matrix=trained_model.decays,
                event_times=processed_data['valid']['buckets'][i]
            )
            for i, seq in tqdm(enumerate(processed_data['valid']['sequences']['time_seqs']),
                desc="Processing valid set",
                total=len(processed_data['valid']['sequences']['time_seqs']),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        ],
        'test': [
            calculator.compute_w(
                baseline=trained_model.baseline,
                t_points=seq,
                adjacency_matrix=trained_model.adjacency,
                decay_matrix=trained_model.decays,
                event_times=processed_data['test']['buckets'][i]
            )
            for i, seq in tqdm(enumerate(processed_data['test']['sequences']['time_seqs']),
                desc="Processing test set",
                total=len(processed_data['test']['sequences']['time_seqs']),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        ]
    }

    all_weights = []
    all_weights.extend(dataset_weights['train'])
    all_weights.extend(dataset_weights['valid'])
    all_weights.extend(dataset_weights['test'])

    # Result analysis
    total_ratio, ratios = WeightAnalyzer.compute_zero_ratio(all_weights)
    print(WeightAnalyzer.format_result(total_ratio, [
        np.mean(np.concatenate(dataset_weights['train']) == 0),
        np.mean(np.concatenate(dataset_weights['valid']) == 0),
        np.mean(np.concatenate(dataset_weights['test']) == 0)
    ]))