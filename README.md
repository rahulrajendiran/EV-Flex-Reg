# ⚡ EV Flexible Regulation Forecast Dashboard

A minimalistic and intelligent **Streamlit web dashboard** that predicts and visualizes **flexible power regulation (kW)** in Electric Vehicle (EV) charging systems.  
This system uses **machine learning and probabilistic forecasting** to estimate how much energy EVs can contribute back to the grid — enabling smarter, cleaner, and more resilient energy networks.

---

## 🌍 Overview

The global shift toward electric mobility has increased the need for **smart energy management**.  
This project demonstrates how **EV charging data** can be transformed into **predictive insights** for **Vehicle-to-Grid (V2G)** flexibility using advanced forecasting techniques.

---

## 🧠 Core Features

### 🔹 Point Forecast (LightGBM)
- Learns complex patterns between energy consumption, duration, and time features.
- Provides **high-accuracy deterministic predictions** of flexible kW.
- Ideal for quick, direct forecasts with minimal uncertainty.

### 🔹 Probabilistic Forecast (Quantile Regression)
- Predicts **Q10, Q50, and Q90** quantiles of flexible kW using Gradient Boosting.
- Captures **forecast uncertainty** — helping grid operators plan for both conservative and optimistic energy return scenarios.

### 🔹 Manual Forecast Interface
- Enter parameters such as:
  - Charging Start/End Time
  - Energy Consumed
  - Duration
  - Day of Week
  - Weekend Indicator  
- Instantly obtain predicted **flexible regulation power**.

### 🔹 Interactive Visualization
- True vs Predicted comparison
- Confidence intervals (10–90%)
- Downloadable CSV results

---

## 🧩 System Workflow

### **Developer Side**
1. **Data Preprocessing**  
   Cleans, transforms, and enriches charging session data (duration, day type, start hour, etc.).  
   Generates the `processed_ev_data.csv` file.

2. **Model Training**  
   - LightGBM → deterministic point forecast  
   - Quantile Regression → probabilistic intervals  
   Models are saved under the `/models/` directory.

3. **Dashboard Development**  
   Built in **Streamlit** with modular structure and responsive layout.  
   Integrates both forecasting and manual entry capabilities.

4. **Deployment**  
   Hosted on **Streamlit Cloud** or any cloud platform (e.g., AWS, Render, Hugging Face Spaces).

### **User Side**
1. Upload preprocessed EV data or use the demo dataset.  
2. Select forecast type → *LightGBM* or *Probabilistic (Q10–Q90)*.  
3. View real-time charts and prediction intervals.  
4. Optionally, enter manual inputs for single-session forecasts.  
5. Download results as CSV for further analysis.

---

## 🧮 Technical Stack

| Layer | Technology | Purpose |
|-------|-------------|----------|
| **Frontend** | Streamlit | Interactive UI and visualization |
| **Backend** | Python | Core logic and computation |
| **ML Models** | LightGBM, GradientBoostingRegressor | Forecasting engine |
| **Libraries** | pandas, numpy, matplotlib, joblib | Data manipulation, plotting, model storage |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EV-Flexible-Regulation-Forecast.git
cd EV-Flexible-Regulation-Forecast

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
