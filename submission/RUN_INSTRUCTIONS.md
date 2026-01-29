# Run Instructions

## Overview

This notebook generates a 24-week (168-day) retail demand forecast at the store-SKU level.

---

## Requirements

### Python Version
- Python 3.8+ (tested on 3.10)

### Dependencies
```
pandas>=1.5.0
numpy>=1.23.0
lightgbm>=3.3.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

Install via:
```bash
pip install pandas numpy lightgbm scikit-learn matplotlib seaborn
```

### Hardware
- **Minimum**: 16GB RAM (for sampling mode)
- **Recommended**: 32GB+ RAM (for full dataset)
- **GPU**: Not required (LightGBM uses CPU)

---

## Data Setup

### Required Files

1. **Sales data** (`final_data 2.csv` or `sales.csv`)
   - Columns: `item_id, store_id, sales, date`
   - ~61 million rows

2. **SKU attributes** (`sku_list_attribute.csv`)
   - Columns: `sku_id, local_imported_attribute`
   - ~3,800 rows

### File Placement

**Option A: Local (Colab or GCP with upload)**
```
/content/data/final_data 2.csv
/content/data/sku_list_attribute.csv
```

**Option B: Google Cloud Storage**
```
gs://your-bucket/data/final_data 2.csv
gs://your-bucket/data/sku_list_attribute.csv
```

Update the `DATA_PATH` and `SKU_ATTR_PATH` variables in the config cell.

---

## Running the Notebook

### Option 1: Google Colab (Recommended for quick start)

1. Upload `RG_Forecasting_Submission.ipynb` to Colab
2. Upload data files to `/content/data/` or mount Google Drive
3. Update config cell paths
4. Run all cells (`Runtime → Run all`)

### Option 2: GCP Vertex AI Workbench

1. Create a Workbench instance (n1-highmem-16 or larger)
2. Clone/upload the notebook
3. Upload data to the instance or use GCS paths
4. Run all cells

### Option 3: Local Jupyter

1. Install dependencies: `pip install -r requirements.txt`
2. Place data files in project directory
3. Update config paths
4. Run: `jupyter notebook RG_Forecasting_Submission.ipynb`

---

## Configuration

Edit the config cell at the top of the notebook:

```python
# === CONFIGURATION ===
DATA_PATH = "data/final_data 2.csv"           # Raw sales data
SKU_ATTR_PATH = "data/sku_list_attribute.csv" # SKU attributes (local/import)
OUTPUT_PATH = "outputs/forecast_168day.csv"   # Forecast output

CUTOFF_DATE = "2025-12-17"      # Last date of training data
FORECAST_START = "2025-12-18"   # First forecast date
HORIZON_DAYS = 168              # 24 weeks

RANDOM_SEED = 42                # For reproducibility
SAMPLE_MODE = False             # Set True for quick testing (uses 10% of data)
```

---

## Output

### Forecast CSV
Location: `outputs/forecast_168day.csv`

| Column | Type | Description |
|--------|------|-------------|
| `item_id` | int | SKU identifier |
| `store_id` | int | Store identifier |
| `date` | str | Forecast date (YYYY-MM-DD) |
| `predicted_sales` | float | Predicted daily sales quantity |

### Expected Statistics
- Rows: ~19 million (114,000 series × 168 days)
- Date range: 2025-12-18 to 2026-06-03
- No negative values
- No NaN values

---

## Runtime Estimates

| Stage | Full Data | Sample Mode (10%) |
|-------|-----------|-------------------|
| Data loading | 2-3 min | 20 sec |
| Panel/spine creation | 5-10 min | 1 min |
| Feature engineering | 10-15 min | 2 min |
| Model training | 15-20 min | 3 min |
| Forecast generation | 5-10 min | 1 min |
| **Total** | **40-60 min** | **~8 min** |

*Times based on n1-highmem-16 (104GB RAM)*

---

## Troubleshooting

### Out of Memory
- Enable `SAMPLE_MODE = True` for testing
- Use a larger machine (n1-highmem-32)
- Process tiers separately

### File Not Found
- Check `DATA_PATH` and `SKU_ATTR_PATH` are correct
- Ensure files are uploaded/accessible

### Slow Training
- LightGBM uses all CPU cores by default
- For faster training, use fewer `n_estimators` (at cost of accuracy)

---

## Verification Checklist

After running, verify:

- [ ] `forecast_168day.csv` exists in output path
- [ ] Row count = number of series × 168
- [ ] Date range is exactly 2025-12-18 to 2026-06-03
- [ ] No negative predictions
- [ ] No NaN predictions
- [ ] Sample plots look reasonable

---

## Contact

For questions about this submission, please refer to `SUBMISSION_SUMMARY.md` for methodology details.
