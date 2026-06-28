# Deployment Guide

## Local run

1. Install the Python dependencies.
2. Generate the trained model artifact.
3. Start the FastAPI app.
4. Open the site in your browser and upload `Airline_review.csv`.
5. If your machine is on Python 3.14, create a Python 3.11 virtual environment first because the scientific packages are much easier to install there.

```powershell
pip install -r requirements.txt
python train_model.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Render free deployment

1. Push the repository to GitHub.
2. Create a new Render Web Service from the repo.
3. Let Render detect `render.yaml`.
4. Render will use `runtime.txt`, which is pinned to Python 3.11.11.
5. Deploy with the free plan.
6. After deploy, open the Render URL and upload the CSV from the site.

## What to do on your side

1. Push the updated files to GitHub.
2. Connect the repo to Render.
3. Confirm the build completes and the model artifact is generated.
4. Visit the live site and test with `Airline_review.csv`.