# Biometric Authentication using Smartphone Sensors

A comprehensive biometric authentication system using accelerometer and gyroscope data from smartphones for user identification and verification.

## Contributors

- **Kunal** 

## Project Overview

This project implements a biometric authentication system that uses motion sensor data (accelerometer and gyroscope) from Samsung Galaxy S6 smartphones to identify and verify users based on their unique gait patterns and movement characteristics.

The system implements both:
- **User Identification**: Multi-class classification to identify which user is using the device
- **User Verification**: Binary classification to verify if a claimed identity is genuine

## Dataset

- **Device**: Samsung Galaxy S6 (Pocket Phone configuration)
- **Sensors**: 6-axis motion data
  - 3-axis Accelerometer (X, Y, Z)
  - 3-axis Gyroscope (X, Y, Z)
- **Users**: 10 unique subjects (IDs: 2, 13, 23, 24, 49, 52, 55, 70, 78, 112)
- **Sample Rate**: Resampled to 100 Hz (10ms intervals)
- **Window Size**: 500 samples (5 seconds) with 50% overlap
- **Total Windows**: 14,588 windowed samples

## Project Structure

```
Biometric Assignment/
├── data/
│   ├── sample_data/          # Raw accelerometer and gyroscope CSV files
│   ├── processed_data/       # Merged 6-channel sensor data
│   ├── outputs/              # Model outputs and metrics
│   │   ├── confusion_matrix.npy
│   │   ├── oof_pred.npy
│   │   ├── oof_proba.npy
│   │   ├── oof_true.npy
│   │   ├── label_classes.npy
│   │   ├── classwise_f1_oof.csv
│   │   ├── fnn_verification_metrics.csv
│   │   └── lstm_verification_metrics.csv
│   ├── X.npy                 # Raw windowed features
│   ├── X_clean.npy           # Cleaned windowed features
│   └── y.npy                 # Labels
├── Assignment1.ipynb         # Main implementation notebook
└── README.md
```

## Methodology

### 1. Data Preprocessing

#### 1.1 Data Loading and Merging
- Load separate accelerometer and gyroscope CSV files
- Merge sensor streams into unified 6-channel time series
- Handle timestamp alignment and synchronization

#### 1.2 Resampling
- Resample all sensors to consistent 100 Hz frequency
- Interpolate missing values to maintain data continuity
- Ensure temporal alignment across all channels

#### 1.3 Windowing
- Window size: 500 samples (5 seconds at 100 Hz)
- Overlap: 250 samples (50% overlap)
- Preserves temporal dynamics while providing sufficient data

#### 1.4 Data Cleaning
- **Z-score clipping**: Remove outliers beyond 3 standard deviations
- **Moving average smoothing**: Window size of 5 samples
- Reduces noise while preserving signal characteristics

### 2. Feature Extraction

Comprehensive feature engineering extracts 96 features (16 features × 6 channels):

#### Statistical Features (60 features)
- Mean, Standard Deviation, Variance
- Median, 25th/75th Percentiles, IQR
- Min, Max, Range
- Mean Absolute Deviation (MAD)

#### Signal Characteristics (12 features)
- Energy: Mean squared amplitude
- RMS (Root Mean Square)

#### Temporal Features (6 features)
- Zero Crossing Rate: Sign change frequency

#### Higher-Order Moments (12 features)
- Skewness: Distribution asymmetry
- Kurtosis: Distribution tail heaviness

#### Frequency Domain Features (6 features)
- Dominant Frequency Index
- Spectral Energy

#### Feature Selection
- SelectKBest with ANOVA F-test
- Reduced to top 10 most discriminative features
- Selected indices: [1, 2, 19, 20, 32, 38, 50, 56, 59, 92]

### 3. Models

#### 3.1 Feed-Forward Neural Network (FNN) - User Identification
**Architecture:**
- Input: 10 selected features
- Hidden Layers: 2 layers × 5 neurons each
- Activation: ReLU
- Output: 10 classes (Softmax)
- Dropout: Applied for regularization

**Training:**
- 5-Fold Stratified Cross-Validation
- Optimizer: Adam
- Max Iterations: 500
- Class weighting for imbalance

**Performance:**
- **Mean Accuracy**: 97.09% ± 0.45%
- **Macro F1-Score**: 95.70% ± 0.69%
- **Weighted F1-Score**: 97.10% ± 0.46%

#### 3.2 LSTM Network - User Identification
**Architecture (Assignment Constraints: ≤6 neurons per layer):**
- Input: (500 timesteps, 6 channels)
- LSTM Layers: 3 layers × 6 units each
- Dropout: 0.2 per layer
- Output: 10 classes (Softmax)
- Total Parameters: 1,006

**Training:**
- 5-Fold Stratified Cross-Validation
- Optimizer: Adam (learning rate: 0.001)
- Epochs: 30 (with early stopping)
- Batch Size: 64
- Class-balanced weighting
- ReduceLROnPlateau callback

**Performance:**
- **OOF Accuracy**: 79.91%
- **Macro AUC**: 0.9664
- **Micro AUC**: 0.9704
- **Macro F1-Score**: 71.63%
- **Weighted F1-Score**: 80.15%

### 4. User Verification (One-vs-All)

#### 4.1 FNN Verification Model
**Architecture:**
- Input: 10 features (same as identification)
- Hidden Layers: 3 layers × 6 neurons each
- Activation: ReLU
- Output: Binary (Sigmoid)
- Dropout: 0.2

**Training:**
- 5-Fold Stratified CV per user (genuine vs impostor)
- Binary cross-entropy loss
- Class-balanced weighting
- Epochs: 40, Batch Size: 128

**Performance (Mean across 10 users):**
- **Mean AUC**: 99.52% ± 0.24%
- **Mean EER**: 1.24% ± 0.65%
- **Mean FAR@EER**: 1.23% ± 0.64%
- **Mean FRR@EER**: 1.24% ± 0.65%

#### 4.2 LSTM Verification Model
**Architecture:**
- Input: (500 timesteps, 6 channels)
- LSTM Layers: 3 layers × 6 units each
- Dropout: 0.2
- Output: Binary (Sigmoid)

**Training:**
- 3-Fold Stratified CV per user
- Binary cross-entropy loss
- Class-balanced weighting
- Epochs: 10, Batch Size: 64

**Performance (Mean across 10 users):**
- **Mean AUC**: 99.63% ± 0.36%
- **Mean EER**: 2.20% ± 1.33%
- **Mean FAR@EER**: 2.18% ± 1.32%
- **Mean FRR@EER**: 2.21% ± 1.34%

## Metrics Explained

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall

### Verification Metrics
- **AUC (Area Under Curve)**: Overall discriminative ability (0-1, higher is better)
- **EER (Equal Error Rate)**: Error rate where FAR = FRR (lower is better)
- **FAR (False Accept Rate)**: Impostor acceptance rate
- **FRR (False Reject Rate)**: Genuine rejection rate

## Key Findings

1. **FNN Identification**: Achieved 97% accuracy with only 10 features, demonstrating effective feature engineering
2. **LSTM Identification**: Achieved 80% accuracy, capturing temporal dynamics in raw sensor data
3. **FNN Verification**: Excellent performance with ~1.24% EER, suitable for real-world deployment
4. **LSTM Verification**: Strong performance with ~2.20% EER, effective temporal pattern recognition
5. **Best User (FNN)**: User 49 achieved 99.51% F1-score
6. **Challenging Users**: Users 2 and 52 showed lower performance, possibly due to less distinctive patterns

## Requirements

```bash
# Python packages
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
scipy
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Biometric Assignment"

# Install dependencies
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn scipy

# Ensure Jupyter is installed
pip install jupyter
```

## Usage

1. **Data Preparation**:
   ```python
   # Run cells 1-18 in Assignment1.ipynb
   # This will process raw sensor data and create X_clean.npy and y.npy
   ```

2. **User Identification (FNN)**:
   ```python
   # Run cells 19-36
   # Trains FNN classifier with feature selection
   ```

3. **User Identification (LSTM)**:
   ```python
   # Run cells 37-52
   # Trains LSTM on raw windowed sequences
   ```

4. **User Verification (FNN)**:
   ```python
   # Run cells 53-56
   # One-vs-all binary classification per user
   ```

5. **User Verification (LSTM)**:
   ```python
   # Run cells 57-58
   # LSTM-based verification per user
   ```

## Results Visualization

The notebooks generate various visualizations:
- Training/validation loss curves
- Confusion matrices
- Class-wise F1-score comparisons
- Per-user verification metrics (AUC, EER)
- ROC curves
- FAR vs FRR trade-offs

## Future Improvements

1. **Model Enhancements**:
   - Experiment with attention mechanisms
   - Try CNN-LSTM hybrid architectures
   - Implement transfer learning

2. **Feature Engineering**:
   - Add frequency domain features (FFT coefficients)
   - Implement wavelet transforms
   - Extract gait-specific features

3. **Data Augmentation**:
   - Time warping
   - Magnitude scaling
   - Noise injection

4. **Real-World Deployment**:
   - Model quantization for mobile devices
   - Real-time inference optimization
   - Continuous authentication framework

## References

- Dataset based on smartphone sensor-based gait recognition research
- Stratified K-Fold Cross-Validation methodology
- LSTM networks for temporal sequence modeling
- Biometric verification standard metrics (EER, FAR, FRR)

## License

This project is part of CS228 coursework.

## Acknowledgments

Special thanks to:
- Course instructors and TAs
- Dataset providers
- Open-source community for ML frameworks

---

**Last Updated**: December 2024

For questions or collaboration, please reach out to the contributors.

