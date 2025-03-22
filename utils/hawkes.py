from typing import Tuple, Optional, Dict, Union, List
import numpy as np
import os
import torch
from torch import Tensor
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
        self._validate_init_params()

    def _validate_init_params(self):
        """Validate initialization parameters"""
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be integer or None, got {type(self.random_seed)}")


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
    
    def compute_intensities(
        self,
        time_seqs: Tensor,
        time_delta_seqs: Tensor,
        type_seqs: Tensor,
        seq_mask: Tensor,
        event_times: List[List[List[float]]],
        n_samples: int = 30
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        
        """Compute Hawkes process intensities with multiple sampling strategies"""
        self._validate_model_initialized()

        # event event intensities
        event_intensities = self._compute_event_intensities(time_seqs[:,:-1], event_times)
        
        # Time delta sampling
        sampled_dtimes = self.sample_time_intervals(time_delta_seqs[:, 1:], n_samples)
        boundary_samples = self.sample_time_intervals(time_delta_seqs[:, :-1], 5)
        
        # Intensity calculations
        sampled_intensities = self._compute_sampled_intensities(sampled_dtimes, event_times)
        boundary_intensities = self._compute_sampled_intensities(boundary_samples, event_times)
        
        return {
            'event_intensities': event_intensities,
            'sampled_intensities': sampled_intensities,
            'processed_tensors': {
                'time_seq': time_seqs,
                'time_delta_seq': time_delta_seqs,
                'type_seq': type_seqs,
                'seq_mask': seq_mask,
                'sample_time_delta_seq': sampled_dtimes,
                'dtime_for_bound_sampled': boundary_samples,
                'bound_sampled_intensities': boundary_intensities
            }
        }   
    
    def _validate_model_initialized(self):
        """Ensure model parameters are available"""
        if not all([hasattr(self.model, attr) for attr in ['baseline', 'adjacency', 'decays']]):
            raise RuntimeError("Model parameters not initialized. Train model first.")

    def _compute_event_intensities(self, time_seqs: Tensor, event_times: List[List[List[float]]]) -> Tensor:
        """Exact intensity computation at event times"""
        batch_size, seq_len = time_seqs.shape
        num_types = len(self.model.baseline)
        intensities = torch.zeros((batch_size, seq_len, num_types))
        
        baseline = torch.tensor(self.model.baseline)
        adjacency = torch.tensor(self.model.adjacency)
        decays = torch.tensor(self.model.decays)

        for b in range(batch_size):
            for t in range(seq_len):
                current_time = time_seqs[b, t].item()
                for k in range(num_types):               
                    intensity = baseline[k].item()
                    for j in range(num_types):
                        for t_j in event_times[b][j]:
                            if t_j <= current_time:
                                delta = current_time - t_j
                                intensity += adjacency[k, j].item() * np.exp(-decays[k, j].item() * delta)
                    intensities[b, t, k] = intensity
        return intensities


    def sample_time_intervals(self, time_delta_seqs: Tensor, n_samples: int) -> Tensor:
        """Generate uniform samples within time intervals"""
        ratios = torch.linspace(0, 1, n_samples)
        return time_delta_seqs[:, :, None] * ratios  # [B, T-1, S]

    def _compute_sampled_intensities(self, sampled_dtimes: Tensor, event_times: List[List[List[float]]]) -> Tensor:
        """Intensity computation at sampled time points"""
        batch_size, seq_len, n_samples = sampled_dtimes.shape
        num_types = len(self.model.baseline)
        intensities = torch.zeros(batch_size, seq_len, n_samples, num_types)
        
        baseline = torch.tensor(self.model.baseline)
        adjacency = torch.tensor(self.model.adjacency)
        decays = torch.tensor(self.model.decays)

        for b in range(batch_size):
            for t in range(seq_len):
                for s in range(n_samples):
                    current_time = sampled_dtimes[b, t, s].item()
                    for k in range(num_types):
                        intensity = baseline[k].item()
                        for j in range(num_types):
                            for t_j in event_times[b][j]:
                                if t_j <= current_time:
                                    delta = current_time - t_j
                                    intensity += adjacency[k, j].item() * np.exp(-decays[k, j].item() * delta)
                        intensities[b, t, s, k] = intensity
        return intensities

def ensure_directory(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

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
    train = processed_data['train']['buckets']
    valid = processed_data['valid']['buckets']
    test = processed_data['test']['buckets']
    trained_model = handler.train_model(model, train)
    
    # 4. Evaluate the model
    datasets = {
        'train': train,
        'valid': valid,
        'test': test
    }
    scores = handler.evaluate_model(trained_model, datasets)

    handler.model = trained_model
    
    # # Output evaluation results
    # print(scores)
    # print(trained_model.baseline, trained_model.adjacency, trained_model.decays)

    train_data = processed_data['train']['sequences']
    valid_data = processed_data['valid']['sequences']
    test_data = processed_data['test']['sequences']
    max_length = max(max(len(seq) for seq in train_data['time_delta_seqs']), max(len(seq) for seq in valid_data['time_delta_seqs']), max(len(seq) for seq in test_data['time_delta_seqs']))
    # print(max_length)

    time_seq = np.full((len(test_data['time_seqs']), max_length), num_event_types, dtype=float)
    time_delta_seq = np.full((len(test_data['time_delta_seqs']), max_length), num_event_types, dtype=float)
    type_seq = np.full((len(test_data['time_delta_seqs']), max_length), num_event_types)
    seq_mask = np.zeros((len(test_data['time_delta_seqs']), max_length), dtype=bool)

    for i, (time_seq_i, time_delta_seq_i, type_seq_i) in enumerate(zip(test_data['time_seqs'], test_data['time_delta_seqs'], test_data['type_seqs'])):
        time_seq[i, :len(time_seq_i)] = time_seq_i
        time_delta_seq[i, :len(time_delta_seq_i)] = time_delta_seq_i
        type_seq[i, :len(type_seq_i)] = type_seq_i
        seq_mask[i, :len(time_seq_i)] = 1

    time_seq = torch.tensor(time_seq, dtype=torch.float32)
    time_delta_seq = torch.tensor(time_delta_seq, dtype=torch.float32)
    type_seq = torch.tensor(type_seq, dtype=torch.long)
    seq_mask = torch.tensor(seq_mask, dtype=torch.bool)    

    # print("Padded Data:\n", time_seq, time_seq.shape, time_delta_seq, time_delta_seq.shape)
    # print("Type Sequence Tensor:\n", type_seq, type_seq.shape)
    # print("Sequence Mask:\n", seq_mask)

    # Compute intensities for the Hawkes process
    intensity_data_dict = handler.compute_intensities(time_seq, time_delta_seq, type_seq, seq_mask, test)

    # print(intensity_data_dict['event_intensities'].shape, intensity_data_dict['event_intensities'])
    # print(intensity_data_dict['sampled_intensities'].shape, intensity_data_dict['sampled_intensities'])
    # print(intensity_data_dict['processed_tensors'])

    # Save results
    save_paths = [r"example\intensity\earthquake\Hawkes\event_intensities.pth",
                  r"example\intensity\earthquake\Hawkes\sample_intensities.pth",
                  r"example\residual_data\earthquake\tensors.pth"]

    for path in save_paths:
        ensure_directory(path)
    
    torch.save(intensity_data_dict['event_intensities'], save_paths[0])
    torch.save(intensity_data_dict['sampled_intensities'], save_paths[1])
    torch.save(intensity_data_dict['processed_tensors'], save_paths[2])