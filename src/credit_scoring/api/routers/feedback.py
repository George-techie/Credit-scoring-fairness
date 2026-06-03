"""/feedback — record ground-truth outcomes for later retraining."""

import sqlite3

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..schemas import FeedbackData

router = APIRouter()
DB_NAME = "feedback.db"


@router.post("/feedback")
def receive_feedback(feedback: FeedbackData):
    try:
        conn = sqlite3.connect(DB_NAME)
        pd.DataFrame([feedback.model_dump()]).to_sql(
            "feedback_v2", conn, if_exists="append", index=False
        )
        conn.close()
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")
