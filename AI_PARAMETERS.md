# AI/ML Parameters Documentation

This document lists all AI and Machine Learning related parameters used in the project.

## 📊 Data Preprocessing Parameters

### Dataset Loading
- **File Path (data_training.py)**: `train_test_network.csv`
- **File Path (RL_training.py)**: `Processed_datasets/Processed_Windows_dataset/windows10_dataset.csv`
- **CSV Engine**: `python` (for RL_training.py)

### Data Cleaning
- **Missing Value Strategy**: `'median'` (SimpleImputer)
- **Data Scaling**: `StandardScaler()` - Standardization (mean=0, std=1)
- **Normalization Range**: `[0, 1]` (min-max scaling for RL environment)

### Train/Test Split
- **Test Size**: `0.2` (20% of data for testing)
- **Random State**: `42` (for reproducibility)
- **Shuffle**: Default (True)

### Feature Selection
- **Excluded Columns**: `['label', 'type']`
- **Data Type Selection**: Numeric only (`select_dtypes(include=[np.number])`)
- **Correlation Analysis**: Absolute correlation coefficients, sorted descending

---

## 🤖 Machine Learning Models Parameters

### 1. Logistic Regression
- **Max Iterations**: `1000`
- **Solver**: Default (usually 'lbfgs')
- **Regularization**: Default

### 2. Random Forest Classifier
- **Parameters**: All default (no explicit hyperparameters set)
  - Default n_estimators: 100
  - Default max_depth: None
  - Default criterion: 'gini'

### 3. Gradient Boosting Classifier
- **Parameters**: All default
  - Default n_estimators: 100
  - Default learning_rate: 0.1
  - Default max_depth: 3

### 4. Ridge Classifier
- **Parameters**: All default
  - Default alpha: 1.0 (regularization strength)

### 5. XGBoost Classifier
- **use_label_encoder**: `False`
- **eval_metric**: `'logloss'`
- **Other parameters**: Default values
  - Default max_depth: 6
  - Default learning_rate: 0.3
  - Default n_estimators: 100

---

## 🎯 Reinforcement Learning Parameters (RL_training.py)

### Environment Configuration
- **Action Space**: `Discrete(2)` - Binary classification (0: normal, 1: alternative/attack)
- **Observation Space**: `Box(low=0, high=1, shape=(num_features,), dtype=np.float32)`
  - Continuous values normalized to [0, 1] range
  - Shape dynamically determined by dataset features

### Reward Function
- **Correct Prediction**: `+1`
- **Incorrect Prediction**: `-1`

### PPO (Proximal Policy Optimization) Agent
- **Policy Type**: `"MlpPolicy"` (Multi-Layer Perceptron)
- **Verbose**: `1` (training progress output)
- **Other PPO Parameters**: Default values
  - Default learning_rate: 3e-4
  - Default n_steps: 2048
  - Default batch_size: 64
  - Default n_epochs: 10
  - Default gamma: 0.99

### Training Parameters
- **Total Timesteps**: `10000`
- **Number of Environments**: `n_envs=1` (single environment vectorized)

### Evaluation Parameters
- **Evaluation Episodes**: `n_eval_episodes=10`
- **Metrics Returned**: `mean_reward`, `std_reward`

---

## 📈 Evaluation Metrics

### Metrics Calculated for All Models
1. **Accuracy Score**: Classification accuracy
2. **Log Loss**: Logarithmic loss (lower is better)
3. **RMSE**: Root Mean Squared Error
4. **R2 Score**: Coefficient of determination
5. **F1 Score**: Harmonic mean of precision and recall
6. **Precision Score**: Precision metric

### Visualization Parameters
- **Figure Size**: `(16, 12)` - 2x3 subplot grid
- **Figure Size (Correlation)**: `(8, 10)`
- **Figure Size (Heatmap)**: `(8, 6)`
- **Heatmap Format**: `".2f"` (2 decimal places)
- **Heatmap Colormap**: `'coolwarm'`
- **Bar Colors**: `['deepskyblue', 'orange', 'limegreen', 'red', 'gold']`
- **Bar Chart Type**: Horizontal (`kind='barh'`)
- **Grid Style**: `linestyle='--', alpha=0.7`

---

## 🔧 Advanced Configuration

### Data Processing
- **Correlation Sort Method**: `"quicksort"`
- **NaN Handling**: Drop rows with NaN values (`dropna()`)
- **Empty String Handling**: Replace with `np.nan`, then convert to numeric

### Environment Reset
- **Reset Strategy**: Return to first data point (`current_step = 0`)

### State Representation
- **State Normalization**: Min-max scaling per feature: `(x - x.min()) / (x.max() - x.min())`
- **Done Condition**: Episode ends when `current_step == len(labels)`

---

## 📝 Summary Table

| Parameter Type | Value | Location |
|---------------|-------|----------|
| Test Split Ratio | 0.2 | data_training.py:119, 166 |
| Random Seed | 42 | data_training.py:119, 166 |
| Imputation Strategy | 'median' | data_training.py:115 |
| Logistic Regression Max Iter | 1000 | data_training.py:144 |
| XGBoost Eval Metric | 'logloss' | data_training.py:197 |
| RL Training Timesteps | 10000 | RL_training.py:71 |
| RL Action Space | Discrete(2) | RL_training.py:34 |
| RL Observation Range | [0, 1] | RL_training.py:36 |
| RL Reward Correct | +1 | RL_training.py:45 |
| RL Reward Incorrect | -1 | RL_training.py:45 |
| RL Evaluation Episodes | 10 | RL_training.py:74 |
| RL Number of Environments | 1 | RL_training.py:65 |

---

## 🎓 Notes

- Most sklearn models use default hyperparameters
- PPO agent uses default stable-baselines3 hyperparameters
- Random state fixed at 42 for reproducibility across runs
- StandardScaler used for feature normalization in ML models
- Min-max scaling used for RL environment observations
- All models evaluated using the same metrics for comparison
