"""Request/response schemas for the API."""

from pydantic import BaseModel


class BorrowerFeatures(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    CNT_CHILDREN: float
    CNT_FAM_MEMBERS: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    FLAG_OWN_CAR: int
    CODE_GENDER: str
    NAME_EDUCATION_TYPE: str
    NAME_INCOME_TYPE: str
    NAME_HOUSING_TYPE: str


class FeedbackData(BorrowerFeatures):
    prediction_prob: float
    ground_truth: int
