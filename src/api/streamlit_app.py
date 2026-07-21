# src/api/streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="IntelliLoan Advisor", page_icon="🏦", layout="wide")

# --- 2. STYLE ---
st.markdown("""
    <style>
    .memo-container {
        background-color: #ffffff; 
        padding: 30px; 
        border-left: 10px solid #003366; 
        border-radius: 8px;
        line-height: 1.8;
        font-family: 'Georgia', serif;
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("System Access")
    api_url = st.text_input("GCP Gateway URL", value="https://intelliloan-api-857396236875.northamerica-northeast1.run.app").strip().rstrip("/")
    api_key = st.text_input("X-API-KEY", type="password")

# --- 4. MAIN ---
st.title("🏦 IntelliLoan: Agentic Credit Appraisal")
client_id = st.number_input("Enter Client ID", value=100002, step=1)

if st.button("🚀 Run Comprehensive Analysis"):
    if not api_key:
        st.error("Please provide the API Key.")
    else:
        endpoint = f"{api_url}/v1/appraise/id/{int(client_id)}"
        
        with st.spinner("Processing..."):
            try:
                response = requests.post(endpoint, headers={"X-API-KEY": api_key}, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # --- 🚀 UNIVERSAL DATA EXTRACTOR (No more 0.0%!) ---
                    # 1. Checking default probability
                    pd_value = None
                    for key in ["probability_of_default", "proba", "pd", "prob", "probability"]:
                        if key in data and data[key] is not None:
                            pd_value = float(data[key])
                            break
                        if "ml_results" in data and key in data["ml_results"]:
                            pd_value = float(data["ml_results"][key])
                            break

                    # 2. If we don't get it, we calculate it based on the credit score
                    score = data.get("credit_score", data.get("ml_results", {}).get("credit_score", 0))
                    if pd_value is None or pd_value == 0.0:
                        pd_value = 1 - (score / 1000)
                    
                    # 3. Extracting other fields
                    verdict = data.get("verdict", data.get("ml_results", {}).get("verdict", "N/A"))
                    risk_lvl = data.get("risk_level", data.get("ml_results", {}).get("risk_level", "N/A"))

                    # --- Display metrics ---
                    st.markdown("### 📊 Decision Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    m1.metric("Credit Score", f"{score}/1000")
                    m2.metric("Verdict", verdict)
                    m3.metric("Risk Level", risk_lvl)
                    # Display pro
                    m4.metric("Prob. of Default", f"{pd_value:.2%}")

                    st.divider()

                    # --- 4. SHAP & MEMO ---
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔍 Local Risk Drivers")
                        drivers = data.get("top_risk_drivers", data.get("shap_reasons", []))
                        if drivers:
                            df_shap = pd.DataFrame(drivers)
                            fig = px.bar(df_shap, x='shap_impact', y='feature', orientation='h', color_continuous_scale='Reds')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No SHAP drivers found.")

                    with col2:
                        st.subheader("📄 AI Compliance Memo")
                        memo = data.get("final_memo", "No memo generated.")
                        # Remplacement des sauts de ligne pour l'affichage HTML
                        formatted_memo = memo.replace('\\n', '<br>').replace('\n', '<br>')
                        st.markdown(f'<div class="memo-container">{formatted_memo}</div>', unsafe_allow_html=True)

                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

st.markdown("---")
st.caption("IntelliLoan AI Platform v0.3.0")