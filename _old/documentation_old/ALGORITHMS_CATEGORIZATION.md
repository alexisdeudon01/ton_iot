# Categorization of Algorithms: Deep Learning vs Light ML

This document categorizes all algorithms used in the project based on their computational complexity and architecture type.

## 🧠 DEEP LEARNING Algorithms

### Neural Network-Based Algorithms

#### 1. **PPO (Proximal Policy Optimization) with MlpPolicy**
- **Location**: `RL_training.py:68`
- **Type**: Reinforcement Learning with Neural Network Policy
- **Architecture**: Multi-Layer Perceptron (MLP) - Deep Neural Network
- **Implementation**: `PPO("MlpPolicy", vec_env, verbose=1)`
- **Characteristics**:
  - Uses deep neural networks as policy/value functions
  - Multiple hidden layers (default PPO architecture)
  - Requires GPU for optimal performance (though can run on CPU)
  - Training: 10,000 timesteps
  - **Computational Cost**: HIGH ⚠️
  - **Memory Usage**: HIGH ⚠️

#### 2. **MLP (Multilayer Perceptrons)** - *Mentioned in README*
- **Location**: Mentioned in README.md (line 26, 47)
- **Type**: Deep Learning Classification Model
- **Status**: ⚠️ **Not in current code** (results shown but implementation not visible)
- **Performance**: Accuracy: 99.99% (from README results)
- **Characteristics**:
  - Multiple hidden layers (deep neural network)
  - Requires TensorFlow/Keras or PyTorch
  - **Computational Cost**: HIGH ⚠️

#### 3. **DQN (Deep Q-Network)** - *Mentioned in README*
- **Location**: Mentioned in README.md (line 26)
- **Type**: Deep Reinforcement Learning
- **Status**: ⚠️ **Not in current code** (mentioned but not implemented)
- **Characteristics**:
  - Deep neural network for Q-value approximation
  - Requires significant computational resources
  - **Computational Cost**: HIGH ⚠️

---

## 🚀 LIGHT / TRADITIONAL ML Algorithms

### These algorithms are fast, interpretable, and have low computational requirements.

#### 1. **Logistic Regression**
- **Location**: `data_training.py:144`
- **Type**: Linear Classification
- **Implementation**: `LogisticRegression(max_iter=1000)`
- **Library**: scikit-learn
- **Characteristics**:
  - Linear model with sigmoid activation
  - Fast training and prediction
  - Interpretable (feature coefficients)
  - **Computational Cost**: LOW ✅
  - **Memory Usage**: LOW ✅
  - **Training Time**: Seconds
  - **Accuracy**: 86.4%

#### 2. **Ridge Classifier**
- **Location**: `data_training.py:173`
- **Type**: Regularized Linear Classification
- **Implementation**: `RidgeClassifier()`
- **Library**: scikit-learn
- **Characteristics**:
  - L2 regularization (prevents overfitting)
  - Linear model
  - Very fast
  - **Computational Cost**: LOW ✅
  - **Memory Usage**: LOW ✅
  - **Training Time**: Seconds
  - **Accuracy**: 82.3%

#### 3. **Random Forest Classifier**
- **Location**: `data_training.py:145`
- **Type**: Ensemble Learning (Tree-based)
- **Implementation**: `RandomForestClassifier()`
- **Library**: scikit-learn
- **Characteristics**:
  - Multiple decision trees (ensemble)
  - Can be parallelized
  - Feature importance available
  - **Computational Cost**: MEDIUM ⚡
  - **Memory Usage**: MEDIUM ⚡
  - **Training Time**: Minutes (depending on data size)
  - **Accuracy**: 99.85% (Best performing!)

#### 4. **Gradient Boosting Classifier**
- **Location**: `data_training.py:146`
- **Type**: Ensemble Learning (Boosting)
- **Implementation**: `GradientBoostingClassifier()`
- **Library**: scikit-learn
- **Characteristics**:
  - Sequential tree building (boosting)
  - More accurate than single trees
  - Slower than Random Forest
  - **Computational Cost**: MEDIUM ⚡
  - **Memory Usage**: MEDIUM ⚡
  - **Training Time**: Minutes to hours
  - **Accuracy**: 99.34%

#### 5. **XGBoost Classifier**
- **Location**: `data_training.py:197`
- **Type**: Advanced Gradient Boosting
- **Implementation**: `XGBClassifier(use_label_encoder=False, eval_metric='logloss')`
- **Library**: XGBoost
- **Characteristics**:
  - Optimized gradient boosting
  - Handles missing values
  - Can use GPU acceleration (optional)
  - **Computational Cost**: MEDIUM-HIGH ⚡⚠️
  - **Memory Usage**: MEDIUM ⚡
  - **Training Time**: Minutes
  - **Accuracy**: 99.85% (Best performing alongside Random Forest!)

#### 6. **SVM (Support Vector Machine)**
- **Location**: `data_training.py:137` (imported but **NOT USED**)
- **Type**: Kernel-based Classification
- **Status**: ⚠️ **Imported but not implemented in current code**
- **Characteristics**:
  - Can be slow for large datasets
  - Good for non-linear problems (with kernels)
  - **Computational Cost**: MEDIUM-HIGH ⚡⚠️

---

## 📊 Summary Table

| Algorithm | Type | Computational Cost | Memory | Training Time | Accuracy | In Code? |
|-----------|------|-------------------|--------|---------------|----------|----------|
| **DEEP LEARNING** | | | | | | |
| PPO (MlpPolicy) | RL + Neural Net | 🔴 HIGH | 🔴 HIGH | Hours | Variable | ✅ Yes |
| MLP | Neural Network | 🔴 HIGH | 🔴 HIGH | Hours | 99.99% | ❌ No* |
| DQN | Deep RL | 🔴 HIGH | 🔴 HIGH | Hours | Variable | ❌ No* |
| **LIGHT ML** | | | | | | |
| Logistic Regression | Linear | 🟢 LOW | 🟢 LOW | Seconds | 86.4% | ✅ Yes |
| Ridge Classifier | Linear | 🟢 LOW | 🟢 LOW | Seconds | 82.3% | ✅ Yes |
| Random Forest | Ensemble | 🟡 MEDIUM | 🟡 MEDIUM | Minutes | **99.85%** | ✅ Yes |
| Gradient Boosting | Ensemble | 🟡 MEDIUM | 🟡 MEDIUM | Minutes | 99.34% | ✅ Yes |
| XGBoost | Advanced Ensemble | 🟡 MEDIUM | 🟡 MEDIUM | Minutes | **99.85%** | ✅ Yes |
| SVM | Kernel-based | 🟡 MEDIUM | 🟡 MEDIUM | Minutes-Hours | - | ❌ Imported only |

\* *Mentioned in README/results but implementation not in current code files*

---

## 🎯 Recommendations

### For Fast Inference / Production:
- ✅ **Random Forest** or **XGBoost** - Best accuracy (99.85%) with reasonable speed
- ✅ **Logistic Regression** - Fastest, acceptable accuracy (86.4%)

### For Maximum Accuracy:
- ✅ **MLP** (if implemented) - 99.99% accuracy (requires GPU)
- ✅ **Random Forest / XGBoost** - 99.85% accuracy (CPU-friendly)

### For Reinforcement Learning:
- ✅ **PPO with MlpPolicy** - Deep RL approach (requires more resources)

### Avoid for Large-Scale Production:
- ⚠️ **PPO** - Too slow for real-time inference
- ⚠️ **Deep Learning models** - Require GPU and more memory

---

## 📈 Performance vs Computational Cost Comparison

```
Accuracy
100% |                                    MLP (99.99%)
     |                    RF/XGBoost (99.85%)
     |                    GB (99.34%)
     |
 80% |        LR (86.4%)
     |    Ridge (82.3%)
     |
     |___|___|___|___|___|___|___|___|___|___|___|___|___|___|___
     LOW      MEDIUM    HIGH        Computational Cost

Legend:
✅ Green: Light algorithms (fast, CPU-friendly)
⚡ Yellow: Medium complexity (parallelizable, optimized)
🔴 Red: Deep learning (requires GPU, high memory)
```

---

## 🔍 Key Insights

1. **Best Light Algorithm**: Random Forest and XGBoost achieve **99.85% accuracy** - competitive with deep learning!
2. **Fastest Algorithm**: Logistic Regression and Ridge (seconds)
3. **Deep Learning**: Only PPO is implemented; MLP/DQN mentioned but code not present
4. **Production-Ready**: Random Forest or XGBoost offer best balance of accuracy and speed
5. **Trade-off**: Deep learning models (MLP) achieve slightly better accuracy (99.99% vs 99.85%) but require significantly more resources
