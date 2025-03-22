import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from .data_loader import load_raw_data
from .data_process import process_dataset
from .hawkes import HawkesModelHandler
from .weight_compute import WeightCalculator, WeightAnalyzer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualDataProcessor:
    """Processor for creating residual datasets based on weight thresholds"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Value cutoff for keeping residual elements (default: 0.5)
        """
        self.threshold = threshold
        self._validate_threshold()

    def _validate_threshold(self):
        """Ensure threshold is in valid range"""
        if not 0 < self.threshold < 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")

    def calculate_residual_indices(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Calculate indices to keep based on weight thresholds
        
        Args:
            weights: List of weight arrays for a dataset
            
        Returns:
            List of index arrays specifying elements to retain
        """
        indices = []
        for weight_array in weights:
            if len(weight_array) < 2:
                indices.append(np.array([0]))
                continue

            # Always keep first and last elements
            keep_indices = [0, len(weight_array)-1]
            
            # Find elements below threshold between first and last
            middle_indices = np.where(weight_array[1:-1] < self.threshold)[0] + 1
            indices.append(np.unique(np.concatenate([keep_indices, middle_indices])))
            
        return indices

    def filter_dataset(self, 
                      raw_data: Dict[str, Any], 
                      indices: List[np.ndarray]) -> Dict[str, List]:
        """
        Filter dataset sequences using calculated indices
        
        Args:
            raw_data: Original dataset dictionary
            indices: List of index arrays to apply
            
        Returns:
            Filtered dataset dictionary
        """
        return {
            'time_seqs': [
                [raw_data['time_seqs'][i][j] for j in idx] 
                for i, idx in enumerate(indices)
            ],
            'type_seqs': [
                [raw_data['type_seqs'][i][j] for j in idx] 
                for i, idx in enumerate(indices)
            ],
            'time_delta_seqs': [
                [raw_data['time_delta_seqs'][i][j] for j in idx] 
                for i, idx in enumerate(indices)
            ]
        }

    def convert_to_serializable_format(self, 
                                      filtered_data: Dict[str, List], 
                                      dim_process: int) -> Dict[str, Any]:
        """
        Convert filtered data to final serialization format
        
        Args:
            filtered_data: Filtered dataset from filter_dataset()
            dim_process: Number of event types
            
        Returns:
            Dictionary ready for pickle serialization
        """
        return {
            'dim_process': dim_process,
            'events': [
                [
                    {
                        'idx_event': idx+1,
                        'type_event': event_type,
                        'time_since_start': time,
                        'time_since_last_event': delta
                    }
                    for idx, (event_type, time, delta) in enumerate(zip(
                        filtered_data['type_seqs'][i],
                        filtered_data['time_seqs'][i],
                        filtered_data['time_delta_seqs'][i]
                    ))
                ]
                for i in range(len(filtered_data['time_seqs']))
            ]
        }

    def process_and_save(self,
                        weights: Dict[str, List[np.ndarray]],
                        raw_datasets: Dict[str, Dict],
                        output_dir: Path,
                        dim_process: int):
        """
        Full processing pipeline with file saving
        
        Args:
            weights: Dictionary containing train/valid/test weights
            raw_datasets: Original datasets dictionary
            output_dir: Path for output files
            dim_process: Number of event types in the data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'valid', 'test']:
            logger.info(f"Processing {split} dataset...")
            
            # Calculate residual indices
            indices = self.calculate_residual_indices(weights[split])
            
            # Filter dataset
            filtered = self.filter_dataset(raw_datasets[split], indices)
            
            # Convert format
            processed = self.convert_to_serializable_format(filtered, dim_process)
            
            # Save results
            output_path = output_dir / f"{split}.pkl"
            with output_path.open('wb') as f:
                pickle.dump(processed, f)
            logger.info(f"Saved {split} data to {output_path}")

if __name__ == "__main__":
    
    # Example usage
    processor = ResidualDataProcessor(threshold=0.5)
    
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

    # Process and save
    output_directory = Path("example/residual_data/earthquake")
    processor.process_and_save(
        weights=dataset_weights,
        raw_datasets={'train': processed_data['train']['sequences'], 
                      'valid': processed_data['valid']['sequences'], 
                      'test': processed_data['test']['sequences']},
        output_dir=output_directory,
        dim_process=num_event_types
    )