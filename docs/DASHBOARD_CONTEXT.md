# RG-Forecasting Dashboard Context

## Quick Start for New Claude Session
In new terminal, run:
```bash
cd /Users/srikavya/Documents/Claude-Projects/RG-Forecasting
claude
```
Then say: "Read docs/DASHBOARD_CONTEXT.md - help me polish the Streamlit dashboard"

---

## Project Overview
- **Retail demand forecasting** pipeline dashboard
- **Streamlit app** at `pipeline_ui/app.py`
- Shows methodology, results, feature importance, accuracy metrics

## Current Dashboard State
- Basic working dashboard exists
- Needs polish: better visuals, narrative flow, charts

## Key Data Sources (all JSON, read-only)
```
/tmp/accuracy_levels/results.json          # WFA at 6 aggregation levels
/tmp/business_metrics/all_tiers_business_metrics.json  # ABC segment metrics
/params/pipeline_params.yaml               # Pipeline configuration
/tmp/spike_analysis/spike_features_summary.json  # Spike feature documentation
```

## Production Accuracy Numbers (for display)
| Metric | T1_MATURE | T2_GROWING |
|--------|-----------|------------|
| Daily WFA | 51.5% | 45.6% |
| Weekly SKU-Store WFA | 56.8% | 67.5% |
| Weekly Store WFA | 87.9% | 80.1% |
| Weekly Total WFA | 88.3% | 80.1% |

## ABC Segment Performance (T1)
- A-items: 58.2% daily WFA (80% of sales)
- B-items: 40.3% daily WFA (15% of sales)
- C-items: 15.1% daily WFA (5% of sales)

## Key Files
- `pipeline_ui/app.py` - Main Streamlit dashboard (~1200 lines)
- `pipeline_ui/requirements.txt` - Dependencies
- `docs/methodology_documentation.md` - All methodology details

## What Needs Polish
1. **Visual improvements**: Add Plotly charts for key metrics
2. **Narrative flow**: Story-driven presentation of methodology
3. **Accuracy funnel**: Show how accuracy improves with aggregation
4. **Feature importance**: Visualize top features per segment
5. **Spike features section**: Document the new inferred promo features

## Existing Plan (from earlier session)
There's a plan file at: `/Users/srikavya/.claude/plans/zippy-sprouting-pillow.md`
This outlines 15 pages with specific enhancements.

## Run Dashboard Locally
```bash
cd pipeline_ui
streamlit run app.py
# Opens at http://localhost:8501
```

## Tech Stack
- Streamlit
- Plotly (for charts)
- Pandas
- YAML (for config)
