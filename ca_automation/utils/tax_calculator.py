# utils/tax_calculator.py
import pandas as pd
import requests

def calculate_gst(df):
    try:
        income = df[df["Category"] == "income"]["Amount"].sum()
        return income * 0.18
    except Exception as e:
        print(f"Error in calculate_gst: {e}")
        return 0.0

def calculate_tds(df):
    try:
        professional_services = df[df["Description"].str.contains("consulting|service", case=False, na=False)]["Amount"].sum()
        return professional_services * 0.10
    except Exception as e:
        print(f"Error in calculate_tds: {e}")
        return 0.0

def file_gst_return(df):
    try:
        response = requests.post("https://api.gst.gov.in/file", json={"data": df.to_dict()})
        return response.json()
    except Exception as e:
        print(f"Error in file_gst_return: {e}")
        return {"status": "error"}