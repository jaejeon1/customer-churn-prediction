#!/bin/bash
# ChurnSense — Start both FastAPI and Streamlit
cd "$(dirname "$0")"
source venv/bin/activate
echo "✅ Starting FastAPI on http://localhost:8000"
uvicorn main:app --reload &
sleep 3
echo "✅ Starting Streamlit on http://localhost:8501"
streamlit run app.py
