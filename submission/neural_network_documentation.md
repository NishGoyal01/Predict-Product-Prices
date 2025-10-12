# Amazon ML Challenge 2025 - Neural Network Solution Documentation

## Team: Hidden Layers
**Members**: Yash Maheshwari (Leader), Aaryaman Patti, Raj Badlani

---

## Methodology

### Problem Understanding
- **Task**: Predict optimal product prices using catalog content for Amazon products
- **Challenge**: Text-based regression with semantic understanding of product descriptions
- **Evaluation**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Target**: Achieve SMAPE < 15% through advanced feature engineering and deep learning

### Approach Overview
Our neural network solution employs a comprehensive text analysis pipeline:
1. **Advanced feature engineering** from catalog content using proven extraction methods
2. **Deep neural architecture** with regularization and batch normalization
3. **Semantic text understanding** through TF-IDF vectorization and statistical features

---

## Model Architecture

### 1. Feature Engineering Pipeline
- **Text Preprocessing**: Advanced cleaning, tokenization, and normalization
- **Brand Extraction**: Pattern-based extraction of 20 top brands (Apple, Samsung, Sony, etc.)
- **IPQ Processing**: Item Pack Quantity extraction using regex patterns for bulk pricing
- **Statistical Features**: Text length, word count, character density, price mentions

### 2. Neural Network Architecture
```
Input (829 features) → Dense(1024) → BatchNorm → Dropout(0.3)
                   → Dense(512)  → BatchNorm → Dropout(0.3)
                   → Dense(256)  → BatchNorm → Dropout(0.2)
                   → Dense(128)  → Dropout(0.2)
                   → Dense(64)   → Dropout(0.1)
                   → Dense(1)    → Linear Output
```

### 3. Training Configuration
- **Optimizer**: Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Loss Function**: Mean Squared Error on log-transformed prices
- **Regularization**: L2 kernel regularization (0.001), Dropout layers
- **Callbacks**: Early stopping (patience=15), Learning rate reduction (factor=0.5)

---

## Feature Engineering

### Text Features (10 features)
- Text length, word count, character density, average word length
- Digit count and ratio, uppercase count and ratio
- Price mentions, technical specification mentions

### Brand Features (13 features)  
- Binary encoding for top 20 brands identified from catalog content
- Brand frequency scores, brand presence indicators
- Brand name length features

### IPQ Features (6 features)
- Raw IPQ values, log-transformed IPQ, clipped IPQ (0-100)
- Binary indicators: has_ipq, is_single_item, is_bulk_item

### TF-IDF Features (800 features)
- Unigram and bigram TF-IDF vectorization
- Stop words removal, min_df=2, max_df=0.95
- Optimized for neural network processing

---

## Model Selection & Training

### Training Process
- **Dataset Split**: 80% train (60,000), 20% validation (15,000)
- **Batch Size**: 256 samples per batch for optimal GPU utilization
- **Epochs**: Early stopping at epoch 97, total training time: 149.6 seconds
- **Target Transformation**: Log(1+price) for handling price skewness

### Validation Strategy
- **Price Range Analysis**: Separate SMAPE calculation for low (<$20), mid ($20-$100), high (>$100) price ranges
- **Cross-validation**: Train/validation split with stratified sampling
- **Performance Monitoring**: Real-time loss and MAE tracking

---

## Technical Implementation

### Libraries & Frameworks
- **Deep Learning**: TensorFlow/Keras with GPU acceleration (T4)
- **Feature Engineering**: scikit-learn (TfidfVectorizer, StandardScaler)
- **Data Processing**: pandas, numpy for efficient data manipulation
- **Evaluation**: Custom SMAPE implementation for competition metric

### Infrastructure
- **Platform**: Google Colab with T4 GPU acceleration
- **Memory Management**: GPU memory growth enabled, batch processing
- **Model Persistence**: HDF5 format for model saving and loading

---

## Results & Performance

| Metric | Performance | Details |
|--------|------------|---------|
| **Overall SMAPE** | **70.13%** | Validation set performance |
| Low prices (<$20) | 62.03% | Good performance on budget items |
| Mid prices ($20-$100) | 78.57% | Moderate performance on mid-range |
| High prices (>$100) | 160.47% | Challenges with luxury/expensive items |
| **vs Baseline** | **-11.15%** | Underperformed XGBoost baseline (58.98%) |

### Key Insights
- **Neural networks struggled** with tabular/structured text features compared to tree-based models
- **Tree-based superiority**: XGBoost better suited for engineered features and sparse TF-IDF
- **High-price prediction difficulty**: Model struggled with expensive items, suggesting need for specialized handling
- **Feature engineering effectiveness**: 829 features successfully created but better suited for traditional ML

---

## Challenges & Lessons Learned

### Technical Challenges
- **Neural network limitations**: Dense layers not optimal for sparse TF-IDF features
- **Overfitting prevention**: Extensive regularization still couldn't match tree-based performance
- **Price range imbalance**: High-price items (>$100) significantly harder to predict accurately

### Model Architecture Insights
- **Tabular data bias**: Confirmed that XGBoost/tree-based models excel on structured features
- **Feature interaction**: Tree models better capture feature interactions automatically
- **Text representation**: TF-IDF sparse features may need different neural architectures (e.g., embedding layers)

---

## Future Improvements

1. **Hybrid Architecture**: Combine embeddings with tree-based models
2. **Advanced Text Models**: Use transformer-based embeddings (BERT/RoBERTa)
3. **Ensemble Approach**: Weight neural network with XGBoost predictions
4. **Price Segmentation**: Separate models for different price ranges
5. **Feature Selection**: Reduce dimensionality while preserving signal

---

## Submission Details

- **Model Type**: Deep Neural Network with comprehensive feature engineering
- **Final Architecture**: 6-layer dense network with 1.55M parameters
- **Training Time**: 149.6 seconds on T4 GPU
- **Validation SMAPE**: 70.13%
- **Submission Format**: 75,000 predictions in required sample_id,price format

**Key Takeaway**: While neural network provided valuable insights into feature effectiveness, tree-based models (XGBoost) remain superior for this tabular text prediction task. This submission establishes a baseline for ensemble approaches and validates our feature engineering pipeline.

---

**Final Model Performance**: SMAPE = 70.13% on validation set
**Competition Context**: Baseline for ensemble development and feature validation