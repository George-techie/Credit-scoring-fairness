"""FastAPI application factory.

Thin transport layer: all scoring/feature logic lives in the importable
``credit_scoring`` package; routers are split by resource.
"""

import os
import sqlite3
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI

from . import state
from .routers import explain, feedback, predict

DB_NAME = "feedback.db"
MODEL_PATH = "model/lgb_model.joblib"


def _load_model() -> None:
    if not os.path.exists(MODEL_PATH):
        return
    artefact = joblib.load(MODEL_PATH)
    if isinstance(artefact, dict) and "model" in artefact:
        state.model = artefact["model"]
        state.feature_names = artefact.get("feature_names")
    else:
        state.model = artefact


def _init_db() -> None:
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            AMT_INCOME_TOTAL REAL, AMT_CREDIT REAL, AMT_ANNUITY REAL,
            AMT_GOODS_PRICE REAL, DAYS_BIRTH REAL, DAYS_EMPLOYED REAL,
            CNT_CHILDREN REAL, CNT_FAM_MEMBERS REAL, EXT_SOURCE_1 REAL,
            EXT_SOURCE_2 REAL, EXT_SOURCE_3 REAL, FLAG_OWN_CAR INTEGER,
            CODE_GENDER TEXT, NAME_EDUCATION_TYPE TEXT, NAME_INCOME_TYPE TEXT,
            NAME_HOUSING_TYPE TEXT, prediction_prob REAL, ground_truth INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    _init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Credit Scoring API", lifespan=lifespan)
    app.include_router(predict.router)
    app.include_router(feedback.router)
    app.include_router(explain.router)
    return app


app = create_app()
