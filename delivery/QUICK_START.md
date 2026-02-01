# Quick Start Guide

Get the RG-Forecast pipeline running in 5 steps.

---

## Step 1: Install Requirements

```bash
# Navigate to project directory
cd /path/to/RG-Forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r delivery/requirements.txt
```

**Minimum Python version**: 3.9+

---

## Step 2: Set Up GCP Credentials

The pipeline uses Google Cloud for BigQuery and Cloud Storage.

```bash
# Option A: Service account key file
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Option B: gcloud CLI authentication
gcloud auth application-default login
```

**Required GCP APIs**:
- BigQuery API
- Cloud Storage API
- Vertex AI API (for training)

**Required IAM Roles**:
- BigQuery Data Editor
- Storage Object Admin
- Vertex AI User (for training)

---

## Step 3: Run the Pipeline

### Full Training Pipeline

```bash
# Train all tiers (T1, T2, T3) with all segments (A, B, C)
python src/train_all_tiers.py
```

**Expected runtime**: 2-4 hours on n2-standard-8

### Validation Only

```bash
# Run cross-validation without production training
python src/validate_all_folds.py
```

### Generate Forecasts

```bash
# Generate 168-day production forecast
python src/generate_168day_forecast.py
```

**Output**: 17.9 million rows saved to BigQuery

---

## Step 4: View Results in Streamlit

```bash
# Launch the dashboard
streamlit run pipeline_ui/app.py
```

**Default URL**: http://localhost:8501

### Dashboard Sections

| Tab | Description |
|-----|-------------|
| Overview | Key metrics and accuracy summary |
| Data Exploration | Raw data analysis |
| Feature Engineering | Feature importance |
| Tiering | T1/T2/T3 distribution |
| Evaluation | Accuracy by segment |
| Production Forecast | Download forecasts |

---

## Step 5: Access Forecasts

### Option A: Streamlit Download

Navigate to "Production Forecast" tab and use download buttons:
- Weekly Store Aggregation
- Weekly SKU-Store
- Daily Store
- A-Items Daily

### Option B: BigQuery Direct Access

```sql
SELECT
    store_id,
    item_id,
    date,
    predicted_sales,
    lower_bound,
    upper_bound
FROM `rg-forecast.forecasting.production_forecast_168d`
WHERE date BETWEEN '2025-12-18' AND '2026-06-03'
```

### Option C: Export to CSV

```python
from google.cloud import bigquery

client = bigquery.Client()
query = """
    SELECT *
    FROM `rg-forecast.forecasting.production_forecast_168d`
"""
df = client.query(query).to_dataframe()
df.to_csv('forecast_output.csv', index=False)
```

---

## Troubleshooting

### "Permission denied" on GCP
```bash
# Verify authentication
gcloud auth list

# Check project
gcloud config get-value project

# Re-authenticate
gcloud auth application-default login
```

### "Module not found" errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r delivery/requirements.txt
```

### Out of memory during training
```bash
# Reduce batch size in config
# Edit configs/config.yaml: batch_size: 50000 -> 25000

# Or use Vertex AI for cloud training
python src/vertex_training/submit_vertex_job.py
```

### Streamlit connection error
```bash
# Check if port 8501 is available
lsof -i :8501

# Use alternative port
streamlit run pipeline_ui/app.py --server.port 8502
```

---

## Configuration

Key parameters in `configs/config.yaml`:

```yaml
# Data paths
data:
  raw_path: "final_data 2.csv"
  attributes_path: "sku_list_attribute.csv"

# Training settings
training:
  validation_horizon: 168  # days
  n_folds: 3  # for T1 mature

# Model hyperparameters (by segment)
model:
  a_items:
    num_leaves: 255
    n_estimators: 1000
    learning_rate: 0.02
  c_items:
    num_leaves: 31
    n_estimators: 300
    learning_rate: 0.1
    min_data_in_leaf: 100
```

---

## Next Steps

1. Review `TECHNICAL_SUMMARY.md` for methodology details
2. Explore the Streamlit dashboard for interactive analysis
3. Check `README.md` for improvement opportunities
