# Residual TPP

This is the official repository of the Residual TPP paper: "[Residual TPP: A Unified Lightweight Approach for Event Stream Data Analysis](https://proceedings.mlr.press/v267/yuan25a.html)".
##  Introduction
Residual TPP is a novel, unified, and lightweight approach for analyzing event stream data. It combines **statistical TPPs** with **neural TPPs**, leveraging the **Residual Events Decomposition (RED)** technique to enhance model flexibility and efficiency. 

By integrating a **Hawkes process** for self-exciting patterns and a **neural TPP** for residual modeling, Residual TPP achieves **state-of-the-art performance** while maintaining computational efficiency.

🔹 **Key Features:**
- A **decomposition-based** approach for event stream modeling.
- Combines **Hawkes processes** with **neural TPPs** for robust predictions.
- **Lightweight and scalable**, reducing computational overhead.
- Achieves **higher accuracy** in goodness-of-fit and prediction tasks.

<!-- ## 🏗️ Installation
Ensure you have Python 3.8+ installed. To set up the environment:

```bash
git clone https://github.com/your-username/ResidualTPP.git
cd ResidualTPP
pip install -r requirements.txt
```

Dependencies:
- `numpy`
- `tick`
- `torch`
- `pandas`
- `matplotlib`
- `scipy`

## 🚀 Usage
### 1️⃣ Data Preprocessing
Prepare event stream datasets using:

```python
from data_loader import load_raw_data
from data_process import process_dataset

raw_data = load_raw_data("earthquake")
processed_data = process_dataset("earthquake")
```

### 2️⃣ Model Initialization
```python
from hawkes_model import HawkesModelHandler

handler = HawkesModelHandler(max_iter=500)
decay_matrix = handler.create_decay_matrix(num_event_types=5)
model = handler.initialize_model()
```

### 3️⃣ Training
```python
train_data = processed_data["train"]["buckets"]
trained_model = handler.train_model(model, train_data)
```

### 4️⃣ Evaluation
```python
datasets = {
    "train": train_data,
    "valid": processed_data["valid"]["buckets"],
    "test": processed_data["test"]["buckets"]
}
scores = handler.evaluate_model(trained_model, datasets)
print("Evaluation Scores:", scores)
``` -->

##  Datasets
We evaluate Residual TPP on three constructed synthetic datasets and six real-world event stream datasets: MIMIC-II, Retweet, StackOverflow, Amazon, Earthquake, and Volcano. The real-world datasets are obtained via  ["EasyTPP"](https://github.com/ant-research/EasyTemporalPointProcess) and ["NHP"](https://github.com/hongyuanmei/neurawkes).


<!-- ## 🎯 Model Architecture
Residual TPP consists of three key components:
1. **Hawkes Process**: Captures periodic/self-exciting event dynamics.
2. **Residual Events Decomposition (RED)**: Identifies events that do not fit the statistical model.
3. **Neural TPP**: Models unexplained residual events for enhanced prediction.

**Final Intensity Function:**
\[
\lambda_k(t) = (1 - \alpha) \lambda_k^{(1)}(t) + \lambda_k^{(2)}(t)
\]
where:
- \( \lambda_k^{(1)}(t) \) is the Hawkes process intensity.
- \( \lambda_k^{(2)}(t) \) is the neural TPP intensity.
- \( \alpha \) represents the proportion of residual events.

##  Experimental Results
| Model         | MIMIC-II | Retweet | Earthquake | StackOverflow | Amazon | Volcano |
|--------------|---------|---------|------------|--------------|--------|--------|
| **Hawkes**   | -2.839  | -13.71  | -4.155     | -2.866       | -0.534 | 0.983  |
| **Residual TPP** | **-2.045** | **-3.240** | **-3.689** | **-0.864** | **-1.419** | **-3.548** |

Residual TPP achieves **superior performance** in goodness-of-fit and prediction tasks while reducing training time. -->


## Citation
If you find this repo useful, please cite our paper.
```bibtex
@InProceedings{pmlr-v267-yuan25a,
  title = 	 {Residual {TPP}: A Unified Lightweight Approach for Event Stream Data Analysis},
  author =       {Yuan, Ruoxin and Fang, Guanhua},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {73455--73477},
  year = 	 {2025},
  volume = 	 {267},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v267/yuan25a.html},
}

```

## Acknowledgments
This project uses code from [EasyTemporalPointProcess](https://github.com/ant-research/EasyTemporalPointProcess).
