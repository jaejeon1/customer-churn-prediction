#!/bin/bash
cd ~/Desktop/p5final
source venv/bin/activate

echo "✅ Starting FastAPI..."
uvicorn main:app --reload &

echo "✅ Starting Streamlit..."
streamlit run app.py &

sleep 4

echo "✅ Opening browser tabs..."
open http://localhost:8501
open http://localhost:8000/docs

wait
EOF
