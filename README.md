# Evaluating the Performance of Machine Learning-Based Classification Models for IoT Intrusion Detection (2024 IEEE ORSS)

<a href="https://doi-org.libpublic3.library.isu.edu/10.1109/ORSS62274.2024.10697949"><img src="https://img.shields.io/badge/-IEEE-00629B?&style=for-the-badge&logo=ieee&logoColor=white" /></a> 
<a href="https://www.researchgate.net/publication/384580344_Evaluating_the_Performance_of_Machine_Learning-Based_Classification_Models_for_IoT_Intrusion_Detection"><img src="https://img.shields.io/badge/-ResearchGate-00CCBB?&style=for-the-badge&logo=researchgate&logoColor=white" /></a>

<img src="https://img.shields.io/badge/-Python-3776AB?&style=for-the-badge&logo=Python&logoColor=white" alt="Python Badge" /> <img src="https://img.shields.io/badge/-OpenAI GYM-0081A5?&style=for-the-badge&logo=openaigym&logoColor=white" /> <img src="https://img.shields.io/badge/-TensorFlow-FF6F00?&style=for-the-badge&logo=tensorflow&logoColor=white" /> <img src="https://img.shields.io/badge/-Pandas-150458?&style=for-the-badge&logo=pandas&logoColor=white" /> <img src="https://img.shields.io/badge/-Keras-D00000?&style=for-the-badge&logo=keras&logoColor=white" /> <img src="https://img.shields.io/badge/-Pytorch-EE4C2C?&style=for-the-badge&logo=pytorch&logoColor=white" /> <img src="https://img.shields.io/badge/-scikit--learn-F7931E?&style=for-the-badge&logo=scikitlearn&logoColor=white" /> <img src="https://img.shields.io/badge/-Overleaf-47A141?&style=for-the-badge&logo=overleaf&logoColor=white" />

## Abstract
As the Internet of Things (IoT) continues to expand its footprint across various sectors, including healthcare, industrial automation, and smart homes, the security of these interconnected devices becomes paramount. With the proliferation of IoT devices, the attack surface for potential cybersecurity threats has significantly increased, necessitating the development of efficient Intrusion Detection Systems (IDS). This study embarks on a comprehensive examination of several machine learning algorithms aimed at enhancing the prediction accuracy of IDS within IoT networks. Leveraging the ToN-IoT dataset, we implement and compare the effectiveness of models. The findings reveal that ensemble methods, particularly Random Forest and XGBoost, exhibit superior performance, underscoring their potential for deployment in safeguarding IoT ecosystems against malicious intrusions. <br>
## ToN-IoT Dataset
The Ton-IoT [dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) used is one of the datasets that were collected from a realistic and large-scale designed network by the Australian Defence Force Academy (ADFA).
This is the [link](https://drive.google.com/file/d/1CAdK9IgIr74RvtR60OdJBuiXKy37egWg/view?usp=sharing) to the used dataset.


## System model
The architecture of our solution is described in the figure below and is divided into 4 main steps: data acquisition, data pre-
processing, data manipulation, and algorithms implementation and evaluation.
<br>
![Proposed Framework Architecture.](AI_implementation.png)

## Installation

### Core Dependencies (Required)
```bash
pip install -r requirements-core.txt
```

### Neural Network & Explainability Dependencies (Optional)
For CNN, TabNet, SHAP, and LIME support:
```bash
pip install -r requirements-nn.txt
```

**Note**: For CUDA-enabled PyTorch, follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Running the code

### IRP Research Pipeline (Main Methodology)

This repository implements the methodology described in the IRP research paper: **"AI-Powered Log Analysis for Smarter Threat Detection"**.

To run the complete pipeline and generate all results:

```bash
python main.py
```

**Note**: The `main.py` file is the only Python file at the project root. All other modules are organized in the `src/` directory.

**Command-line options**:
```bash
python main.py                    # Run complete pipeline
python main.py --phase 1          # Run only Phase 1 (preprocessing)
python main.py --phase 3          # Run only Phase 3 (evaluation)
python main.py --phase 5          # Run only Phase 5 (ranking)
python main.py --output-dir custom_output  # Custom output directory
```

This will execute three phases:
1. **Phase 1: Preprocessing Configuration Selection** - Complete preprocessing pipeline with all sub-steps
2. **Phase 3: Multi-Dimensional Algorithm Evaluation** - Evaluation across 3 dimensions (Performance, Resources, Explainability)
3. **Phase 5: AHP-TOPSIS Ranking** - Multi-criteria decision making for algorithm ranking

#### Phase 1: Preprocessing Pipeline - Detailed Sub-steps

The preprocessing pipeline follows a structured workflow with 6 main steps:

**1. Harmonization of Features**
- **CIC-DDoS2019**: Uses 80+ features extracted by CICFlowMeter
- **TON_IoT**: Original features or CIC-ToN-IoT version (if available)
- **Method**: Identifies common features through exact matches and semantic similarity
- **Output**: Harmonized datasets with unified feature schema

**2. Label Alignment (Binary Classification)**
- **CIC-DDoS2019**: 
  - Label column: Last column of CSV
  - Binary mapping: `Benign = 0`, all attacks (non-Benign) = `1`
- **TON_IoT**: 
  - Label column: Last column of CSV
  - Filtering: Keep only rows with `type='normal'` or `type='ddos'`
  - Binary mapping: `normal = 0`, `ddos = 1`
- **Output**: Standardized binary labels (0 = Normal/Benign, 1 = Attack/DDoS)

**3. Unified Preprocessing**

The fused dataset undergoes the following sub-steps:

**3.1. Data Cleaning**
- Remove NaN and Infinity values
- Convert all columns to numeric
- Drop columns that are all NaN
- Impute remaining NaN values with median

**3.2. Encoding**
- Encode categorical features using LabelEncoder
- Handle missing categorical values

**3.3. Feature Selection**
- Select top K features (default: 20) using Mutual Information
- Reduces dimensionality and training time
- Keeps most discriminative features

**3.4. Scaling**
- Normalize features using RobustScaler (median and IQR based)
- Robust to outliers (better than StandardScaler for network traffic data)
- Values typically in range [-3, 3]

**3.5. Resampling**
- Balance classes using SMOTE (Synthetic Minority Over-sampling Technique)
- Prevents model bias towards majority class
- Creates synthetic samples for minority class

**4. Data Splitting (Train/Validation/Test)**

**4.1. Stratified Splitting**
- Split data into 3 parts with stratification:
  - **Training Set**: 70% (default)
  - **Validation Set**: 15% (default)
  - **Test Set**: 15% (default)
- Stratification ensures proportional representation of both datasets (TON_IoT and CIC-DDoS2019) in each split
- Maintains class distribution across splits

**4.2. Cross-Validation**
- 5-fold stratified cross-validation for model evaluation
- Ensures robust performance estimation

#### Three-Dimensional Evaluation Framework

The framework evaluates algorithms across three dimensions:

##### Dimension 1: Detection Performance
- **Metrics**: F1 Score (primary), Precision (Pr), Recall (Rc), Accuracy
- **Calculation**: Using weighted average for multi-class problems (following CIC-DDoS2019 methodology)
- **Formulas**: See `DIMENSIONS_CALCULATION.md` for detailed mathematical formulas

##### Dimension 2: Resource Efficiency
- **Metrics**: Training time (seconds), Memory usage (MB)
- **Calculation**: Normalized combination of time and memory (60% time, 40% memory)
- **Interpretation**: Higher score = more efficient (faster training, less memory)

##### Dimension 3: Explainability
- **Components**: 
  - Native Interpretability (50%): Binary indicator for tree-based models
  - SHAP Score (30%): Mean Absolute SHAP Values
  - LIME Score (20%): Mean importance from LIME explanations
- **Interpretation**: Higher score = more explainable model

**Detailed calculations**: See `DIMENSIONS_CALCULATION.md` for complete formulas and visualizations.

**Visual representations**: 
- Bar charts for each dimension
- Radar/spider charts for combined 3D visualization
- Scatter plots for multi-dimensional comparison

#### Algorithms Evaluated

According to IRP methodology:
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- CNN (Convolutional Neural Network for tabular data)
- TabNet (Deep learning for tabular data)

#### Datasets

- **TON_IoT**: Included (train_test_network.csv)
- **CIC-DDoS2019**: Download from [CIC Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html) and place in `data/raw/CIC-DDoS2019/`

##### CIC-DDoS2019 Dataset

The CIC-DDoS2019 dataset is a comprehensive DDoS attack dataset containing:
- **80 network traffic features** extracted using CICFlowMeter software (publicly available from CIC)
- **11 types of DDoS attacks**: reflective DDoS (DNS, LDAP, MSSQL, TFTP), UDP, UDP-Lag, SYN, and others
- Both benign and attack traffic flows

**Reference Paper**: "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy" by Sharafaldin et al. (2019), Canadian Institute for Cybersecurity (CIC).

**Download**: [CIC-DDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)

**Documentation**: See `dataset.pdf` in the repository for detailed methodology.

##### Using Both Datasets Together

The pipeline uses **both TON_IoT and CIC-DDoS2019 datasets** through a process of harmonization and early fusion:

1. **Harmonization**: Features from both datasets are mapped to a common schema based on semantic similarity (exact matches and semantic mappings)
2. **Early Fusion**: The harmonized datasets are combined into a single dataset with statistical validation (Kolmogorov-Smirnov test)
3. **Preprocessing**: The fused dataset undergoes SMOTE (for class balancing) and RobustScaler (for feature scaling)

**Dataset Structure for CIC-DDoS2019**:

⚠️ **No File Organization Required**: The dataset loader automatically detects and loads CSV files from **any location** within `datasets/cic_ddos2019/`:
- Root directory: `datasets/cic_ddos2019/*.csv`
- Subdirectories: `datasets/cic_ddos2019/*/*.csv` (e.g., `examples/Training-Day01/`)
- Nested subdirectories: `datasets/cic_ddos2019/*/*/*.csv`

**Why you don't need to reorganize:**
1. **Flexible Loading**: The loader recursively searches all subdirectories
2. **Automatic Filtering**: Template files (containing "example", "sample", "template", or "structure" in name) are excluded
3. **No Manual Work**: Files can stay in their current locations (e.g., `examples/Training-Day01/`)

**Legacy paths**: The pipeline also checks `data/raw/CIC-DDoS2019/` for backward compatibility.

**Fallback Behavior**: If CIC-DDoS2019 is not available, the pipeline automatically falls back to using TON_IoT alone.

**See also**: `OUTPUT_EXPECTED.md` for detailed information on dataset processing outputs.

### Legacy Scripts

- `data_training.py` - Data exploration and basic ML models (Legacy code, kept for reference)
- `RL_training.py` - Reinforcement Learning implementation (Optional)
## Results

### IRP Pipeline Results

Results are automatically generated in the `output/` directory:
- `output/phase1_preprocessing/` - Preprocessed datasets, harmonization statistics
- `output/phase3_evaluation/` - 3D evaluation metrics, algorithm reports, visualizations
  - `algorithm_reports/` - Detailed reports per algorithm
  - `visualizations/` - Graphs and charts for each dimension
- `output/phase5_ranking/` - AHP-TOPSIS ranking results, decision matrices
- `output/logs/` - Log files for each phase

**Expected outputs**: See `OUTPUT_EXPECTED.md` for detailed documentation of all output files, their formats, and how to interpret them.

### Data Exploration

First, we explored the data and we deducted a correlation analysis as in the figure below: <br>

![Correlation with the target feature label](results/data_analysis/correlation.png)

The main results are illustrated in the table and figure below:
<br>
![Classifiers Performance.](Models.png)
<br>

### Model Evaluation Metrics

| Model                | Accuracy   | Log Loss       | RMSE      | F1 Score  | Precision Score | R2 Score   |
|----------------------|------------|----------------|-----------|-----------|-----------------|------------|
| Ridge                | 0.822502   | **0.6397665**  | 0.421305  | 0.893667  | 0.822655        | 0.019612   |
| XGBoost              | **0.998484** | 0.054652     | **0.038939** | **0.999006** | 0.998820        | **0.991625** |
| Logistic Regression  | 0.863986   | 0.368590       | 0.368800  | 0.915657  | 0.868558        | 0.248745   |
| Random Forest        | **0.998484** | **0.006544**  | **0.038939** | 0.999006  | **0.998913**     | **0.991625** |
| Gradient Boosting    | 0.993390   | 0.032367       | 0.081302  | 0.995674  | 0.993810        | 0.963491   |

Next, we used the Multilayer Perceptrons (MLP) in Deep Learning and the Deep Q-Network in Reinforcement Learning on the same data, and the performances are as follows: <br>

![](DLAccuracy.png)
![](DLLoss.png)

### Performance Metrics of the MLP Model

| **Metric**   | **Value**   |
|--------------|-------------|
| R2 Score     | 0.9994765   |
| Log Loss     | 0.0002139   |
| Precision    | 1.0         |
| F1 Score     | 0.9999378   |
| RMSE         | 0.0097348   |
| Accuracy     | 0.9999052   |

![](drlperf.png)


## Citation
Please do not hesitate to contribute to this project and cite us:
```
@INPROCEEDINGS{10697949,
  author={Kaddour, Hamza and Das, Shaibal and Bajgai, Rishikesh and Sanchez, Amairanni and Sanchez, Jason and Chiu, Steve C. and Ashour, Ahmed F. and Fouda, Mostafa M.},
  booktitle={2024 IEEE Opportunity Research Scholars Symposium (ORSS)}, 
  title={Evaluating the Performance of Machine Learning-Based Classification Models for IoT Intrusion Detection}, 
  year={2024},
  volume={},
  number={},
  pages={84-87},
  keywords={Machine learning algorithms;Biological system modeling;Ecosystems;Intrusion detection;Smart homes;Predictive models;Data models;Internet of Things;Random forests;Optimization;IoT security;intrusion detection systems;cyber-security threats;IoT networks;malicious intrusions;safeguarding IoT ecosystems},
  doi={10.1109/ORSS62274.2024.10697949}}
```
You can find an extension to this paper where included the DL and RL and comparison in the following [document](https://drive.google.com/file/d/1Z0cyVbdsaaw-EuPd6IidkRskHtIQmqM-/view?usp=sharing). <br>
If you find this project interesting, please do not hesitate to reach out to me for any recommendations, questions, or suggestions.
[Email me](mailto:hamzakaddour@isu.edu)
