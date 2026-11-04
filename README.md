# LITTLE HELPER — class demonstration build

## Included
- `app.py`: Streamlit prototype
- `staffing_data.csv`: 30,000-row demonstration extract with the fields used by the app
- `requirements.txt`: Python dependencies

## Important data note
The supplied source dataset contained 1,303,830 records. The app's existing model-training code already caps training at 30,000 records. This deployment build uses a reproducible random 30,000-row extract to reduce repository size and hosting startup/memory requirements. It is suitable for demonstrating the prototype, not for operational or clinical decisions.

The supplied source `WorkDate` values are YYYYMMDD integers. The app has been corrected to parse them as calendar dates.

Weather, visitor activity, and appointment activity remain illustrative simulations. Staffing estimates are historical-pattern comparisons, not safe-staffing recommendations.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish a public demo with Streamlit Community Cloud
1. Create a new **private or public GitHub repository** (public is simplest for a class demo).
2. Upload `app.py`, `staffing_data.csv`, and `requirements.txt` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository, branch, and `app.py`.
5. Deploy. Streamlit will provide a shareable `*.streamlit.app` URL.
6. Open the URL in a private/incognito browser window and test it before sharing.

Do not commit API keys, passwords, or private resident/facility information. The included dataset is the public U.S. staffing dataset provided for this prototype.
