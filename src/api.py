import json

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from preprocessing import preprocess_data
from utils import deserialize_data


class Item(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float

    def to_dataframe(self):
        return pd.DataFrame([self.model_dump()])


class CleanedItem(BaseModel):
    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    home_ownership_MORTGAGE: float
    home_ownership_OTHER: float
    home_ownership_OWN: float
    home_ownership_RENT: float
    loan_intent_DEBTCONSOLIDATION: float
    loan_intent_EDUCATION: float
    loan_intent_HOMEIMPROVEMENT: float
    loan_intent_MEDICAL: float
    loan_intent_PERSONAL: float
    loan_intent_VENTURE: float
    grade_A: float
    grade_B: float
    grade_C: float
    grade_D: float
    grade_E: float
    grade_F: float
    grade_G: float
    default_on_file_N: float
    default_on_file_Y: float

    def to_dataframe(self):
        return pd.DataFrame([self.model_dump()])


def get_best_threshold():
    with open("../models/best_threshold.json", "r") as file:
        best_threshold = json.load(file)

    return best_threshold["threshold"]


threshold = get_best_threshold()
model = deserialize_data("../models/logreg_base.pkl").best_estimator_


app = FastAPI()


@app.post("/preprocess")
async def preprocess(item: Item):
    x_data = item.to_dataframe()
    x_data_clean = preprocess_data(x_data).iloc[[0]].to_dict(orient="records")[0]
    return x_data_clean


@app.post("/predict")
async def predict(cleaned_item: CleanedItem):
    cleaned_item = cleaned_item.to_dataframe()
    y_probabilities = model.predict_proba(cleaned_item)[:, 1][0]

    if y_probabilities >= threshold:
        return {"prediction": "yes", "probability": y_probabilities}
    else:
        return {"prediction": "no", "probability": y_probabilities}
