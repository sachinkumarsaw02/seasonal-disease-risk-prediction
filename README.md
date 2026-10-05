# Seasonal Disease Risk Prediction — Public Deployment Ready

This is the original academic project with its existing dataset, trained Random Forest pipeline, pages, styling and map. The deployment changes are only the files needed to run the same project locally and on a public web host.

## A. Run on Windows (no VS Code required)

1. Install Python 3.12 from https://www.python.org/downloads/ and enable Add Python to PATH.
2. Open CMD.
3. Change directory to this folder. Example:

    E:\Seasonal_Disease_Risk_Prediction\Seasonal_Disease_Risk_Prediction

4. Install packages:

    py -m pip install -r requirements.txt

5. Start:

    py app.py

6. Open http://127.0.0.1:5000

You can also double-click run_project.bat.

## B. Publish it for anyone

1. Create a GitHub repository named `seasonal-disease-risk-prediction`.
2. Upload the complete contents of this folder (including model/, dataset/, templates/ and static/).
3. On Render, create a New Web Service and connect the GitHub repository.
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Python version is pinned by `.python-version` to 3.12.
7. Deploy. Render will provide an HTTPS URL that can be opened from any device.

## Important

The model and dataset are the original academic project's model and dataset. The dataset is synthetic/educational. The prediction is not a medical diagnosis and should not be used for clinical or public-health decisions.
