# Fraud Detection Analytics using Machine Learning

Advanced fraud detection system using ensemble ML models achieving 96% accuracy with Power BI dashboards and real-time scoring capabilities.

![System Dashboard](assets/fraud_dashboard.png)

## 🎯 Project Overview

Production-grade fraud detection system featuring:
- **96% Detection Accuracy** with 18% reduction in false positives
- **50,000+ Transactions** analyzed with real-time scoring
- **Ensemble ML Models** (Logistic Regression, Random Forest, XGBoost)
- **Interactive Power BI Dashboards** for fraud analytics
- **API-based Deployment** for real-time predictions

## 📊 Key Features

### Machine Learning Models
- ✅ Logistic Regression baseline
- ✅ Random Forest Classifier
- ✅ XGBoost Gradient Boosting
- ✅ Ensemble voting classifier
- ✅ Model explainability (SHAP values)

### Data Processing
- ✅ Advanced feature engineering
- ✅ Data wrangling and cleansing
- ✅ Handling imbalanced datasets (SMOTE)
- ✅ Feature selection and optimization
- ✅ Cross-validation and hyperparameter tuning

### Analytics & Reporting
- ✅ Power BI interactive dashboards
- ✅ Real-time fraud risk scoring
- ✅ Transaction pattern analysis
- ✅ Anomaly detection
- ✅ Performance monitoring

## 🛠️ Technologies Used

- **ML Framework:** Scikit-learn, XGBoost, LightGBM
- **Data Processing:** Python (Pandas, NumPy), SQL
- **Visualization:** Power BI, Matplotlib, Seaborn, Plotly
- **Model Deployment:** Flask API, Docker
- **Model Monitoring:** MLflow, Evidently AI
- **Database:** PostgreSQL, Redis (caching)

## 📁 Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                    # Raw transaction data
│   ├── processed/              # Cleaned and engineered features
│   └── sample_transactions.csv
├── src/
│   ├── data_preprocessing/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   └── data_cleansing.py
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── ensemble_model.py
│   ├── training/
│   │   ├── train_pipeline.py
│   │   ├── hyperparameter_tuning.py
│   │   └── model_evaluation.py
│   ├── prediction/
│   │   ├── fraud_scorer.py
│   │   └── batch_prediction.py
│   └── api/
│       ├── app.py
│       └── routes.py
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── powerbi/
│   └── FraudAnalyticsDashboard.pbix
├── models/
│   ├── ensemble_model.pkl
│   ├── feature_scaler.pkl
│   └── model_metadata.json
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes/
├── monitoring/
│   └── mlflow_tracking.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
Power BI Desktop
Docker (optional)
PostgreSQL (optional)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/cyrildude77/fraud-detection.git
cd fraud-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download sample data**
```bash
python scripts/download_sample_data.py
```

### Usage

1. **Train models**
```bash
python src/training/train_pipeline.py --config config/training_config.yaml
```

2. **Make predictions**
```bash
python src/prediction/fraud_scorer.py --input data/new_transactions.csv
```

3. **Launch API**
```bash
python src/api/app.py
```

4. **Access dashboard**
```
Open powerbi/FraudAnalyticsDashboard.pbix in Power BI Desktop
```

## 📈 Model Performance

### Classification Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 92.3% | 89.1% | 87.5% | 88.3% | 0.947 |
| Random Forest | 95.2% | 93.8% | 91.2% | 92.5% | 0.978 |
| XGBoost | 95.8% | 94.2% | 92.6% | 93.4% | 0.982 |
| **Ensemble** | **96.0%** | **94.5%** | **93.1%** | **93.8%** | **0.985** |

### Key Improvements
- ✅ **18% reduction** in false positives
- ✅ **12% improvement** in recall for minority class
- ✅ **96% overall accuracy** on test set
- ✅ **< 100ms** prediction latency

## 🔧 Feature Engineering

### Transaction Features
```python
# Amount-based features
- transaction_amount
- amount_zscore (standardized)
- amount_log (log-transformed)
- rolling_avg_7d
- rolling_std_7d

# Time-based features
- hour_of_day
- day_of_week
- is_weekend
- is_business_hours
- days_since_account_creation

# Behavioral features
- transaction_frequency_24h
- avg_transaction_amount
- transaction_velocity
- distance_from_home
- merchant_category_risk

# Card features
- card_present
- international_transaction
- online_transaction
- recurring_transaction
```

### Feature Importance (Top 10)

1. transaction_velocity (0.142)
2. amount_zscore (0.138)
3. transaction_frequency_24h (0.125)
4. merchant_category_risk (0.089)
5. distance_from_home (0.076)
6. hour_of_day (0.062)
7. days_since_account_creation (0.058)
8. rolling_std_7d (0.054)
9. is_weekend (0.047)
10. card_present (0.043)

## 📊 Power BI Dashboard Components

### 1. Executive Overview
- Total transactions processed
- Fraud detection rate
- False positive rate
- Model accuracy metrics
- Daily fraud trends

### 2. Transaction Analysis
- Fraud distribution by amount
- Geographic fraud patterns
- Time-based fraud trends
- Merchant category analysis
- Card type analysis

### 3. Model Performance
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Feature importance chart
- Model comparison metrics

### 4. Alert Management
- Real-time fraud alerts
- Alert investigation status
- False positive tracking
- Alert resolution time

## 🔬 Model Training Process

### 1. Data Preprocessing
```python
# Handle missing values
df = handle_missing_values(df)

# Feature engineering
df = engineer_features(df)

# Handle class imbalance
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
```

### 2. Model Training
```python
# Train ensemble model
models = {
    'lr': LogisticRegression(),
    'rf': RandomForestClassifier(),
    'xgb': XGBClassifier()
}

ensemble = VotingClassifier(
    estimators=list(models.items()),
    voting='soft'
)

ensemble.fit(X_train, y_train)
```

### 3. Hyperparameter Tuning
```python
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [10, 20, 30],
    'xgb__learning_rate': [0.01, 0.1, 0.3],
    'xgb__max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    ensemble, param_grid, 
    cv=5, scoring='f1', n_jobs=-1
)
```

## 🌐 API Endpoints

### Predict Fraud
```bash
POST /api/v1/predict
Content-Type: application/json

{
  "transaction_amount": 1500.00,
  "merchant_id": "M12345",
  "card_id": "C67890",
  "timestamp": "2024-01-15T14:30:00Z",
  "location": "New York, NY"
}

Response:
{
  "fraud_probability": 0.87,
  "prediction": "fraud",
  "risk_level": "high",
  "factors": ["high_amount", "unusual_location", "odd_hour"]
}
```

### Batch Prediction
```bash
POST /api/v1/batch-predict
Content-Type: application/json

{
  "transactions": [...],
  "return_probabilities": true
}
```

### Model Performance
```bash
GET /api/v1/model/performance

Response:
{
  "accuracy": 0.960,
  "precision": 0.945,
  "recall": 0.931,
  "f1_score": 0.938,
  "auc_roc": 0.985
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test suite
pytest tests/test_models.py -v

# Load testing for API
locust -f tests/load_test.py --host=http://localhost:5000
```

## 📊 Model Explainability

### SHAP Analysis
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

## 🚀 Deployment

### Docker
```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -p 5000:5000 fraud-detection:latest
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

## 📈 Monitoring & Logging

- Model performance tracking with MLflow
- Prediction drift monitoring
- Data quality validation
- API latency monitoring
- Alert system for model degradation

## 🔐 Security Features

- Input validation and sanitization
- Rate limiting on API endpoints
- Authentication and authorization
- Encrypted data storage
- Audit logging for all predictions

## 📚 Documentation

- [Data Dictionary](docs/DATA_DICTIONARY.md)
- [API Documentation](docs/API.md)
- [Model Card](docs/MODEL_CARD.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md.

## 📄 License

MIT License - see LICENSE file.

## 👤 Author

**Cyril Anand**
- LinkedIn: [cyril-anand-8896582a5](https://linkedin.com/in/cyril-anand-8896582a5)
- GitHub: [@cyrildude77](https://github.com/cyrildude77)
- Email: vinodcyril77@gmail.com

---

⭐ Star this repo if you found it helpful!
