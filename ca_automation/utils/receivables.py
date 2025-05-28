import pandas as pd
from datetime import datetime, timedelta

def aging_analysis(df):
    try:
        df["Date"] = pd.to_datetime(df["Date"])
        today = datetime.now()
        df["Days Overdue"] = (today - df["Date"]).dt.days
        receivables = df[df["Category"] == "income"][["Description", "Amount", "Days Overdue"]]
        return receivables[receivables["Days Overdue"] > 30]
    except Exception as e:
        print(f"Error in aging_analysis: {e}")
        return pd.DataFrame({"Description": [], "Amount": [], "Days Overdue": []})

def send_reminder(df):
    try:
        reminders = []
        for _, row in df.iterrows():
            reminders.append(f"Reminder: Payment of ₹{row['Amount']} for {row['Description']} is {row['Days Overdue']} days overdue.")
        return reminders
    except Exception as e:
        print(f"Error in send_reminder: {e}")
        return []