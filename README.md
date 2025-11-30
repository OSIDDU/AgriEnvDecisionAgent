
# Agri-Env Decision Agent (Agent-Style ML Project)

This project implements an **agent-style AI system** for:

- Crop Yield Prediction
- Air Quality Index (AQI) Prediction

The agent:
- Observes input (soil / pollutant data)
- Calls ML tools (Random Forest models)
- Produces decisions & advisories

## How to Run

```bash
pip install -r requirements.txt

python scripts/preprocess.py
python notebooks/train_models.py

uvicorn api.main:app --reload     # terminal 1
streamlit run dashboard/app.py    # terminal 2
