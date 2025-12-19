# OrbitPay Credit Score Prediction API

FastAPI application for predicting customer credit scores using machine learning with SQLite storage.

## 🏗️ Project Structure

```
orbit-pay/
├── ml_models/                          # Your trained ML models
│   ├── stacked_ensemble.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── target_mapping.pkl
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app entry point
│   ├── config.py                       # Configuration
│   ├── database.py                     # SQLite setup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── prediction.py               # Pydantic models
│   │   └── database_models.py          # SQLAlchemy ORM
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py       # ML logic
│   ├── routers/
│   │   ├── __init__.py
│   │   └── prediction.py               # API endpoints
│   └── utils/
│       ├── __init__.py
│       └── feature_engineering.py      # Feature creation
├── predictions.db                      # SQLite database (auto-created)
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Update Configuration

Edit `app/config.py` to point to your model files:

```python
MODEL_DIR: Path = Path("ml_models")
BEST_MODEL_PATH: Path = MODEL_DIR / "stacked_ensemble.pkl"  # or your best model
```

### 3. Create __init__.py Files

Create empty `__init__.py` files in:
- `app/__init__.py`
- `app/models/__init__.py`
- `app/services/__init__.py`
- `app/routers/__init__.py`
- `app/utils/__init__.py`

### 4. Run the Application

```bash
# From the project root directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. **Health Check**
```http
GET /health
```
Check if the API and models are loaded properly.

### 2. **Single Prediction**
```http
POST /api/v1/predict
```

**Request Body:**
```json
{
  "age": 35,
  "monthly_inhand_salary": 5000.0,
  "num_credit_card": 3,
  "outstanding_debt": 2500.0,
  "credit_utilization_ratio": 35.5,
  "monthly_balance": 1500.0,
  "credit_mix": "Good",
  "customer_id": "CUST001"
}
```

**Response:**
```json
{
  "prediction_id": 1,
  "predicted_score": "Good",
  "confidence": 0.87,
  "probabilities": {
    "Poor": 0.05,
    "Standard": 0.08,
    "Good": 0.87
  },
  "input_data": { ... },
  "timestamp": "2025-01-15T10:30:00",
  "model_version": "1.0.0"
}
```

### 3. **Get Prediction by ID**
```http
GET /api/v1/predictions/{prediction_id}
```

### 4. **Get Customer History**
```http
GET /api/v1/predictions/customer/{customer_id}?limit=10
```

### 5. **Get All Predictions**
```http
GET /api/v1/predictions?skip=0&limit=50&score_filter=Good
```

### 6. **Delete Prediction**
```http
DELETE /api/v1/predictions/{prediction_id}
```

## 💻 Usage Examples

### Python Example

```python
import requests

# API endpoint
url = "http://localhost:8000/api/v1/predict"

# Input data
data = {
    "age": 35,
    "monthly_inhand_salary": 5000.0,
    "num_credit_card": 3,
    "outstanding_debt": 2500.0,
    "credit_utilization_ratio": 35.5,
    "monthly_balance": 1500.0,
    "credit_mix": "Good",
    "customer_id": "CUST001"
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print(f"Predicted Score: {result['predicted_score']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "monthly_inhand_salary": 5000.0,
    "num_credit_card": 3,
    "outstanding_debt": 2500.0,
    "credit_utilization_ratio": 35.5,
    "monthly_balance": 1500.0,
    "credit_mix": "Good",
    "customer_id": "CUST001"
  }'
```

### JavaScript/Fetch Example

```javascript
const url = 'http://localhost:8000/api/v1/predict';

const data = {
  age: 35,
  monthly_inhand_salary: 5000.0,
  num_credit_card: 3,
  outstanding_debt: 2500.0,
  credit_utilization_ratio: 35.5,
  monthly_balance: 1500.0,
  credit_mix: "Good",
  customer_id: "CUST001"
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(result => {
    console.log('Predicted Score:', result.predicted_score);
    console.log('Confidence:', result.confidence);
  });
```

## 🗄️ Database Schema

The SQLite database stores all predictions with the following schema:

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER NOT NULL,
    monthly_inhand_salary REAL NOT NULL,
    num_credit_card INTEGER NOT NULL,
    outstanding_debt REAL NOT NULL,
    credit_utilization_ratio REAL NOT NULL,
    monthly_balance REAL NOT NULL,
    credit_mix VARCHAR(20) NOT NULL,
    predicted_score VARCHAR(20) NOT NULL,
    confidence REAL NOT NULL,
    probabilities JSON NOT NULL,
    customer_id VARCHAR(100),
    model_version VARCHAR(20) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    debt_to_income REAL,
    risk_score REAL
);
```

## 🔍 Viewing Database

You can view the SQLite database using:

1. **SQLite Browser**: Download from https://sqlitebrowser.org/
2. **Command Line**:
```bash
sqlite3 predictions.db
.tables
SELECT * FROM predictions LIMIT 10;
```

3. **Python**:
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('predictions.db')
df = pd.read_sql_query("SELECT * FROM predictions", conn)
print(df)
conn.close()
```

## 🔧 Configuration Options

Edit `app/config.py` to customize:

```python
class Settings(BaseSettings):
    APP_NAME: str = "OrbitPay Credit Score API"
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./predictions.db"
    MODEL_DIR: Path = Path("ml_models")
    MAX_PREDICTIONS_PER_REQUEST: int = 100
```

## 📊 Model Performance

Your trained model achieves:
- **Accuracy**: 85-90%+ (depending on best model)
- **Classes**: Poor, Standard, Good
- **Features**: 40+ engineered features
- **Algorithm**: Stacked Ensemble (XGBoost + LightGBM + RF + GB)

## 🐛 Troubleshooting

### Models Not Loading
```
Error: Model not found at 'ml_models/stacked_ensemble.pkl'
```
**Solution**: Update `MODEL_DIR` and model paths in `app/config.py`

### Import Errors
```
ModuleNotFoundError: No module named 'app'
```
**Solution**: Run from project root: `uvicorn app.main:app --reload`

### Database Errors
```
Error: unable to open database file
```
**Solution**: Ensure write permissions in project directory

## 🚢 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Create `.env` file:
```
DEBUG=False
DATABASE_URL=sqlite:///./database/predictions.db
MODEL_DIR=ml_models
```

## 📝 Testing

```python
# test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict():
    data = {
        "age": 35,
        "monthly_inhand_salary": 5000.0,
        "num_credit_card": 3,
        "outstanding_debt": 2500.0,
        "credit_utilization_ratio": 35.5,
        "monthly_balance": 1500.0,
        "credit_mix": "Good"
    }
    response = client.post("/api/v1/predict", json=data)
    assert response.status_code == 200
    assert "predicted_score" in response.json()
```

Run tests:
```bash
pytest test_api.py
```

## 📚 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **SQLAlchemy Docs**: https://docs.sqlalchemy.org/
- **Pydantic Docs**: https://docs.pydantic.dev/

## 📄 License

MIT License - OrbitPay 2025
