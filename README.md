# Climate Analytical Tool

Streamlit dashboard for analyzing weekly AIED Lab climate survey data.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Survey CSV files are intentionally not committed to GitHub. Upload weekly CSV files from the Streamlit sidebar after opening the app.

## Online Sharing

This app is designed for Streamlit hosting. After pushing this repository to GitHub, deploy it with:

- Main file: `app.py`
- Python dependencies: `requirements.txt`

Streamlit Community Cloud or Render are better fits for this app than Vercel because Streamlit runs a persistent Python web server.
