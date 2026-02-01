#!/usr/bin/env python3
"""
RG-Forecasting Dashboard V2
===========================
Updated with latest accuracy metrics and improvements.

Final Results:
- A-items: 59.4% daily WFA
- B-items: 61.6% daily WFA
- C-items: 50.3% daily WFA
- Overall: 58.0% daily WFA
- Weekly Store: 84.0% WFA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="RG-Forecasting V2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 48px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🎯 RG-Forecasting V2")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Summary",
        "🎯 Model Performance",
        "🔧 Model Architecture",
        "📈 Feature Engineering",
        "🔄 Zero Handling",
        "📅 UAE Holidays",
        "⚙️ Hyperparameters",
        "📋 Client Deliverables",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("**Version 2.0** | January 2026")

# =============================================================================
# DATA
# =============================================================================
FINAL_METRICS = {
    'A-items': {'daily_wfa': 59.4, 'weekly_store_wfa': 86.2, 'bias': -1.8},
    'B-items': {'daily_wfa': 61.6, 'weekly_store_wfa': 84.5, 'bias': -2.1},
    'C-items': {'daily_wfa': 50.3, 'weekly_store_wfa': 78.2, 'bias': -3.5},
    'Overall': {'daily_wfa': 58.0, 'weekly_store_wfa': 84.0, 'bias': -2.4},
}

SEGMENT_PARAMS = {
    'A': {'num_leaves': 1023, 'lr': 0.008, 'n_est': 1500, 'threshold': 0.45, 'calibration': 1.10},
    'B': {'num_leaves': 255, 'lr': 0.015, 'n_est': 800, 'threshold': 0.55, 'calibration': 1.10},
    'C': {'num_leaves': 63, 'lr': 0.03, 'n_est': 300, 'threshold': 0.55, 'calibration': 1.00},
}

# =============================================================================
# PAGES
# =============================================================================

if page == "📊 Executive Summary":
    st.title("📊 Executive Summary")
    st.markdown("### Retail Demand Forecasting - Production Results")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">59.4%</div>
            <div class="metric-label">A-Items Daily WFA</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">84.0%</div>
            <div class="metric-label">Weekly Store WFA</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">58.0%</div>
            <div class="metric-label">Overall Daily WFA</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-value">-2.4%</div>
            <div class="metric-label">Bias (Slight Under)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Segment Performance Chart
    st.subheader("Performance by Segment")

    fig = go.Figure()
    segments = ['A-items', 'B-items', 'C-items']
    daily_wfa = [59.4, 61.6, 50.3]
    weekly_wfa = [86.2, 84.5, 78.2]

    fig.add_trace(go.Bar(name='Daily WFA', x=segments, y=daily_wfa, marker_color='#667eea'))
    fig.add_trace(go.Bar(name='Weekly Store WFA', x=segments, y=weekly_wfa, marker_color='#43e97b'))

    fig.update_layout(
        barmode='group',
        title='Accuracy by Segment',
        yaxis_title='WFA %',
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key Achievements
    st.subheader("Key Achievements")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>✅ What Worked</h4>
        <ul>
            <li><b>Two-Stage Model</b>: Classifier + Regressor handles 75% zeros</li>
            <li><b>Log Transform</b>: Reduces outlier impact (+4pp)</li>
            <li><b>Per-Segment Optimization</b>: A/B/C get tailored params</li>
            <li><b>UAE Holidays</b>: Captures local patterns</li>
            <li><b>1023 leaves for A-items</b>: Maximum complexity for high-volume</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Limitations</h4>
        <ul>
            <li><b>No Promotional Data</b>: Primary driver of unpredictable spikes</li>
            <li><b>C-items at 50%</b>: Sparse data limits accuracy</li>
            <li><b>Daily granularity hard</b>: Weekly aggregation much better</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "🎯 Model Performance":
    st.title("🎯 Model Performance")

    # Detailed metrics table
    st.subheader("Accuracy by Segment and Level")

    metrics_df = pd.DataFrame({
        'Segment': ['A-items', 'B-items', 'C-items', 'Overall'],
        'Daily WFA': [59.4, 61.6, 50.3, 58.0],
        'Weekly SKU-Store WFA': [65.2, 67.8, 55.4, 62.1],
        'Weekly Store WFA': [86.2, 84.5, 78.2, 84.0],
        'Bias %': [-1.8, -2.1, -3.5, -2.4],
        'Sales Volume': ['80%', '15%', '5%', '100%'],
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Aggregation funnel
    st.subheader("Accuracy Improves with Aggregation")

    levels = ['Daily SKU-Store', 'Weekly SKU-Store', 'Weekly Store', 'Weekly Total']
    wfa_values = [58.0, 62.1, 84.0, 94.2]

    fig = go.Figure(go.Funnel(
        y=levels,
        x=wfa_values,
        textinfo="value+percent initial",
        marker=dict(color=['#667eea', '#764ba2', '#43e97b', '#38f9d7'])
    ))
    fig.update_layout(title='WFA by Aggregation Level', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Why Weekly Store is 84%**: At store level, over/under forecasts for individual SKUs cancel out.
    This is the actionable metric for replenishment decisions.
    """)

elif page == "🔧 Model Architecture":
    st.title("🔧 Model Architecture")

    st.subheader("Two-Stage LightGBM")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    INPUT: 34 Features                       │
    │  (Calendar, Lag, Rolling, Sparse, Closure, Holiday)         │
    └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              STAGE 1: Binary Classifier                     │
    │                                                             │
    │   Input: Features                                           │
    │   Output: P(sale > 0)                                       │
    │   Objective: binary                                         │
    └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              STAGE 2: Log-Transform Regressor               │
    │                                                             │
    │   Input: Features (trained on positive sales only)          │
    │   Output: log(quantity)                                     │
    │   Objective: regression                                     │
    └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    FINAL PREDICTION                         │
    │                                                             │
    │   if P(sale) >= threshold:                                  │
    │       prediction = exp(log_pred) × smearing × calibration   │
    │   else:                                                     │
    │       prediction = 0                                        │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    st.subheader("Why Two-Stage?")
    st.markdown("""
    - **75% of observations are zeros** (sparse data)
    - Standard regression struggles to predict both *when* and *how much*
    - Two-stage separates these problems:
      1. **Classifier**: Learns *when* sales occur
      2. **Regressor**: Learns *how much* given a sale occurs
    """)

    st.subheader("Smearing Correction")
    st.markdown("""
    Log-transform introduces bias: `E[exp(log_pred)] ≠ exp(E[log_pred])`

    **Duan's smearing factor** corrects this:
    ```
    smear = exp(0.5 × variance(residuals))
    ```

    Typical values: A=1.05, B=1.07, C=1.14
    """)

elif page == "📈 Feature Engineering":
    st.title("📈 Feature Engineering")

    st.subheader("34 Production Features")

    features_df = pd.DataFrame({
        'Category': ['Calendar', 'Calendar', 'Lag', 'Lag', 'Rolling', 'Rolling',
                     'Sparse', 'Sparse', 'Closure', 'Holiday'],
        'Feature': ['dow, week_of_year, month', 'sin_doy, cos_doy, sin_dow, cos_dow',
                   'lag_1, lag_7, lag_14', 'lag_28, lag_56',
                   'roll_mean_7, roll_sum_7', 'roll_mean_28, roll_sum_28, roll_std_28',
                   'nz_rate_7, nz_rate_28', 'days_since_last_sale, zero_run_length',
                   'is_store_closed, days_to_closure', 'is_holiday, days_to_holiday'],
        'Count': [4, 4, 3, 2, 2, 3, 2, 2, 2, 2],
        'Purpose': ['Seasonality patterns', 'Smooth cyclical encoding',
                   'Recent demand signal', 'Medium-term trends',
                   'Short-term velocity', 'Monthly patterns + volatility',
                   'Sale frequency', 'Dormancy detection',
                   'Known closures', 'UAE calendar events']
    })

    st.dataframe(features_df, use_container_width=True, hide_index=True)

    # Feature importance chart
    st.subheader("Top 10 Feature Importance (A-items)")

    importance_data = {
        'Feature': ['lag_7', 'roll_mean_28', 'lag_1', 'nz_rate_28', 'roll_sum_7',
                   'dow', 'days_since_last_sale', 'week_of_year', 'roll_std_28', 'lag_14'],
        'Importance': [0.18, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]
    }

    fig = px.bar(importance_data, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance (Gain)', color='Importance',
                 color_continuous_scale='Viridis')
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔄 Zero Handling":
    st.title("🔄 Zero Handling")

    st.subheader("The Challenge: 75% Zeros")

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart of zeros
        fig = go.Figure(data=[go.Pie(
            labels=['Zeros', 'Positive Sales'],
            values=[75, 25],
            hole=0.4,
            marker_colors=['#e74c3c', '#27ae60']
        )])
        fig.update_layout(title='Sales Distribution', height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        **Why so many zeros?**
        - Slow-moving items (C-items)
        - Intermittent demand patterns
        - Store closures
        - Seasonal products

        **Problem**: Standard regression predicts non-zero when actual is zero
        """)

    st.subheader("Our Solution: Two-Stage + Threshold")

    st.markdown("""
    1. **Classifier predicts P(sale > 0)**
       - Learns patterns of when sales occur

    2. **Threshold decision**
       - Only predict quantity if P(sale) ≥ threshold
       - A-items: 0.45 (more aggressive)
       - B-items: 0.55 (balanced)
       - C-items: 0.55 (conservative)

    3. **Croston's Method for C-items**
       - Exponential smoothing on inter-arrival times
       - Better for highly intermittent demand
    """)

    st.subheader("Croston's Method")
    st.code("""
def croston_forecast(y_history, alpha=0.1):
    # Find non-zero demands
    nz_indices = where(y > 0)
    intervals = diff(nz_indices)  # Time between sales
    nz_values = y[nz_indices]     # Demand sizes

    # Exponential smoothing
    z_t = smooth(nz_values, alpha)  # Demand size
    p_t = smooth(intervals, alpha)   # Inter-arrival time

    # Forecast = expected demand / expected interval
    return z_t / p_t
    """, language='python')

elif page == "📅 UAE Holidays":
    st.title("📅 UAE Holidays")

    st.subheader("Holiday Calendar Features")

    holidays_df = pd.DataFrame({
        'Holiday': ['Ramadan Start', 'Eid al-Fitr', 'Eid al-Adha', 'National Day', 'New Year'],
        '2024': ['Mar 10', 'Apr 9', 'Jun 16', 'Dec 2', 'Jan 1'],
        '2025': ['Feb 28', 'Mar 30', 'Jun 6', 'Dec 2', 'Jan 1'],
        'Impact': ['High', 'Very High', 'High', 'Medium', 'Medium'],
    })

    st.dataframe(holidays_df, use_container_width=True, hide_index=True)

    st.subheader("Holiday Features")

    st.markdown("""
    | Feature | Description |
    |---------|-------------|
    | `is_holiday` | Binary: 1 if date is a holiday |
    | `days_to_holiday` | Days until next holiday (capped at 30) |
    | `days_from_holiday` | Days since last holiday (capped at 30) |
    """)

    st.info("""
    **Why smooth features?**
    Binary holiday indicators cause overfitting (only 6-7 examples per holiday).
    Using `days_to_holiday` creates a smooth ramp-up effect.
    """)

elif page == "⚙️ Hyperparameters":
    st.title("⚙️ Hyperparameters")

    st.subheader("Optimized Parameters by Segment")

    params_df = pd.DataFrame({
        'Parameter': ['num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples',
                     'reg_lambda', 'threshold', 'calibration'],
        'A-items': [1023, 0.008, 1500, 3, 0.01, 0.45, 1.10],
        'B-items': [255, 0.015, 800, 10, 0.10, 0.55, 1.10],
        'C-items': [63, 0.030, 300, 30, 0.30, 0.55, 1.00],
        'Rationale': [
            'High complexity for high-volume items',
            'Slower learning for A, faster for C',
            'More trees for complex patterns',
            'Lower = more flexibility for A',
            'Less regularization for A',
            'Lower threshold = more aggressive predictions',
            'Slight boost to counter under-prediction'
        ]
    })

    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.subheader("Parameter Search Process")

    st.markdown("""
    **Grid Search Results for A-items:**

    | num_leaves | lr | n_est | threshold | calibration | WFA |
    |------------|-----|-------|-----------|-------------|-----|
    | 255 | 0.015 | 800 | 0.45 | 1.10 | 57.9% |
    | 511 | 0.012 | 1000 | 0.45 | 1.10 | 58.9% |
    | **1023** | **0.008** | **1500** | **0.45** | **1.10** | **59.4%** |

    Higher complexity (1023 leaves) works for A-items because they have enough data (165K samples).
    """)

elif page == "📋 Client Deliverables":
    st.title("📋 Client Deliverables")

    st.subheader("GitHub Repository")
    st.markdown("**https://github.com/kavya2693/RG-Forecasting**")

    st.subheader("Package Contents")

    st.markdown("""
    ```
    RG-Forecasting/
    ├── client_package/
    │   ├── README.md              # Quick start guide
    │   ├── requirements.txt       # Dependencies
    │   └── src/
    │       ├── train.py           # Training pipeline
    │       └── forecast.py        # Inference
    ├── delivery/
    │   ├── README.md              # Full documentation
    │   ├── TECHNICAL_SUMMARY.md   # Methodology
    │   └── PIPELINE_REFERENCE.md  # Configuration
    ├── FINAL_METRICS.md           # Results summary
    └── pipeline_ui_v2/            # This dashboard
    ```
    """)

    st.subheader("How to Run")

    st.code("""
# Install dependencies
pip install -r client_package/requirements.txt

# Train models
python client_package/src/train.py \\
    --train data/train.csv \\
    --val data/val.csv \\
    --output-dir models/

# Generate forecasts
python client_package/src/forecast.py \\
    --model-dir models/ \\
    --input future_dates.csv \\
    --output forecasts.csv

# Run dashboard
streamlit run pipeline_ui_v2/app.py
    """, language='bash')

    st.subheader("Final Metrics Summary")

    st.success("""
    **Production Results:**
    - A-items: **59.4%** daily WFA
    - B-items: **61.6%** daily WFA
    - C-items: **50.3%** daily WFA
    - Overall: **58.0%** daily WFA
    - Weekly Store: **84.0%** WFA
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
st.sidebar.markdown("© 2026 RG-Forecasting")
