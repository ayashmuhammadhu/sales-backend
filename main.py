from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from model import predict_sales

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

stored_df = {}


@app.get("/")
def root():
    return {"status": "Sales Prediction API running"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    required = ["Product line", "Quantity", "Date", "Unit price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}
    stored_df["data"] = df
    products = df["Product line"].unique().tolist()
    return {"products": products}


@app.post("/analyze")
async def analyze(product: str = Query(...)):
    if "data" not in stored_df:
        return {"error": "No data uploaded"}
    df = stored_df["data"]
    df["Date"] = pd.to_datetime(df["Date"])
    product_df = df[df["Product line"] == product]
    daily = product_df.groupby("Date")["Quantity"].sum()
    chart_data = [
        {"date": str(k.date()), "quantity": int(v)}
        for k, v in daily.items()
    ]
    return {"chart_data": chart_data}


@app.post("/predict")
async def predict(
    product: str = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
):
    if "data" not in stored_df:
        return {"error": "No data uploaded"}
    df = stored_df["data"]
    result = predict_sales(df, product, start_date, end_date)
    return {"predicted_quantity": result}
