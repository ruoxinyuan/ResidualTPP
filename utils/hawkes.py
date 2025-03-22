from typing import Tuple, Optional
import numpy as np
from tick.hawkes import HawkesExpKern

from .data_loader import load_raw_data
from .data_process import process_dataset


class HawkesModelHandler:
    def __init__(self, max_iter: int = 500, random_seed: Optional[int] = 42):
        """Initialize the Hawkes process model handler
        
        Args:
            max_iter: The maximum number of iterations, default is 500
            random_seed: The random seed for reproducibility, default is 42 (None disables it)
        """
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.model = None
        self.decay_matrix = None
    
    def create_decay_matrix(
        self,
        num_event_types: int,
        base_value: float = 1.0,
        noise_scale: float = 0.01
    ) -> np.ndarray:
        """Create a decay matrix (identity matrix + Gaussian noise)
        
        Args:
            num_event_types: The number of event types
            base_value: The base value for the diagonal, default is 1.0
            noise_scale: The noise scaling factor, default is 0.01
            
        Returns:
            np.ndarray: Decay matrix of shape (num_event_types, num_event_types)
        """
        if num_event_types <= 0:
            raise ValueError(f"Invalid number of event types num_event_types={num_event_types}, must be a positive integer.")
        if base_value <= 0:
            raise ValueError(f"Base value must be positive, got base_value={base_value}")

        identity = np.eye(num_event_types) * base_value
        noise = noise_scale * np.random.randn(num_event_types, num_event_types)
        self.decay_matrix = identity + noise
        return self.decay_matrix

    def initialize_model(self) -> HawkesExpKern:
        """Initialize the Hawkes process model
        
        Returns:
            HawkesExpKern: The initialized Hawkes model
        """
        if self.decay_matrix is None:
            raise ValueError("Decay matrix has not been created. Please call create_decay_matrix() first.")

        return HawkesExpKern(decays=self.decay_matrix, max_iter=self.max_iter)

    def train_model(self, model: HawkesExpKern, train_data: list) -> HawkesExpKern:
        """Train the Hawkes process model
        
        Args:
            model: The initialized Hawkes model
            train_data: The training data in the format expected by the tick library
            
        Returns:
            HawkesExpKern: The trained Hawkes model
        """
        model.fit(train_data)
        return model

    def evaluate_model(self, model: HawkesExpKern, datasets: dict) -> Tuple[float, float, float]:
        """Evaluate the model on different datasets
        
        Args:
            model: The trained Hawkes model
            datasets: A dictionary containing 'train', 'valid', and 'test' datasets
            
        Returns:
            Tuple: A tuple containing the scores for train, valid, and test datasets
        """
        scores = []
        for split in ['train', 'valid', 'test']:
            data = datasets.get(split)
            if not data:
                raise ValueError(f"Missing {split} dataset")
                
            score = model.score(
                events=data,
                baseline=model.baseline,
                adjacency=model.adjacency
            )
            scores.append(score)
            
        return tuple(scores)


if __name__ == "__main__":
    # Configuration parameters
    MAX_ITER = 500

    # Load raw data
    raw_data = load_raw_data("example/configs/origin_earthquake.yaml", 'earthquake')
    num_event_types = raw_data["train"]["dim_process"]
    
    # Initialize the model handler
    handler = HawkesModelHandler(max_iter=MAX_ITER, random_seed=42)
    
    # 1. Create decay matrix
    decay_matrix = handler.create_decay_matrix(num_event_types, 4, 0.01)
    
    # 2. Initialize the model
    model = handler.initialize_model()
    
    # 3. Train the model
    processed_data = process_dataset("example/configs/origin_earthquake.yaml", "earthquake")
    train_data = processed_data['train']['buckets']
    valid_data = processed_data['valid']['buckets']
    test_data = processed_data['test']['buckets']
    trained_model = handler.train_model(model, train_data)
    
    # 4. Evaluate the model
    datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }
    scores = handler.evaluate_model(trained_model, datasets)
    
    # Output results
    print(scores)
    print(trained_model.baseline, trained_model.adjacency, trained_model.decays)
