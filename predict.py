import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


import uvicorn

import xgboost as xgb


model_file = 'xgb_model.bin'

with open(model_file, 'rb') as f_in:
    model, features_xgb = pickle.load(f_in)


app = FastAPI()

class ExoplanetFeatures(BaseModel):
    pl_orbper: float      # Orbital period (days)
    pl_trandurh: float    # Transit duration (hours)
    pl_trandep: float     # Transit depth (ppm)
    pl_rade: float        # Planet radius (Earth radii)
    pl_insol: float       # Insolation flux (Earth flux)
    pl_eqt: float          # Equilibrium temperature (K)
    st_tmag: float         # TESS magnitude
    st_dist: float         # Stellar distance (pc)
    st_teff: float         # Stellar effective temperature (K)
    st_logg: float          # Stellar surface gravity (log g)
    st_rad: float          # Stellar radius (Solar radii)


fetures_xgb = list(ExoplanetFeatures.model_fields.keys())


@app.post('/predict')
def predict(candidate: ExoplanetFeatures):
    row = pd.DataFrame([candidate.model_dump()])
    X = xgb.DMatrix(row[features_xgb], feature_names = features_xgb)
    xgb_pred = model.predict(X)[0]
    planet = (xgb_pred >= 0.5)


    result  = {
        'likely_planet': bool(planet),
        'planet_probability': f'{float(xgb_pred)  * 100}%'
    }

    return result


