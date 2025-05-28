import pandas as pd
from datetime import datetime
import markdown


def generate_report(df):
   try:
      total_income = df[df["Category"] == "income"]["Amount"].sum()
      total_expenses = df[df["Category"] == "expense"]["Amount"].sum()
      net_profit = total_income - total_expenses

      report = f"""
# Financial Report - {datetime.now().strftime('%Y-%m-%d')}

## Overview
- **Total Income**: ₹{total_income:.2f}
- **Total Expenses**: ₹{total_expenses:.2f}
- **Net Profit**: ₹{net_profit:.2f}

## Insights
- **Top Expense Category**: {df[df["Category"] == "expense"]["Description"].mode()[0] if not df[df["Category"] == "expense"].empty else "N/A"}
- **Largest Income Source**: {df[df["Category"] == "income"]["Description"].mode()[0] if not df[df["Category"] == "income"].empty else "N/A"}

## Recommendations
- Optimize high-expense categories.
- Explore additional income streams based on current trends.
"""
      return markdown.markdown(report)
   except Exception as e:
      print(f"Error in generate_report: {e}")
      return "<p>Error generating report.</p>"


# utils/portfolio.py
import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
   try:
      d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
      d2 = d1 - sigma * np.sqrt(T)
      if option_type == "call":
         price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
      else:
         price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
      return price
   except Exception as e:
      print(f"Error in black_scholes_price: {e}")
      return 0.0


# utils/database.py
import psycopg2
from psycopg2 import pool


class Database:
   _pool = None

   @classmethod
   def init_db(cls):
      try:
         cls._pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,
            user="postgres",
            password="password",
            host="localhost",
            port="5432",
            database="ca_automation"
         )
         with cls._pool.getconn() as conn:
            with conn.cursor() as c:
               c.execute('''CREATE TABLE IF NOT EXISTS clients
                                 (id SERIAL PRIMARY KEY,
                                  name TEXT)''')
               c.execute('''CREATE TABLE IF NOT EXISTS transactions
                                 (id SERIAL PRIMARY KEY,
                                  description TEXT,
                                  amount REAL,
                                  category TEXT,
                                  date TEXT,
                                  client_id INTEGER REFERENCES clients(id))''')
               conn.commit()
      except Exception as e:
         print(f"Database initialization error: {e}")

   @classmethod
   def get_conn(cls):
      return cls._pool.getconn()

   @classmethod
   def release_conn(cls, conn):
      cls._pool.putconn(conn)


def init_db():
   Database.init_db()


def add_client(name):
   try:
      conn = Database.get_conn()
      with conn.cursor() as c:
         c.execute("INSERT INTO clients (name) VALUES (%s)", (name,))
         conn.commit()
      Database.release_conn(conn)
   except Exception as e:
      print(f"Error adding client: {e}")


def get_clients():
   try:
      conn = Database.get_conn()
      with conn.cursor() as c:
         c.execute("SELECT id, name FROM clients")
         clients = c.fetchall()
      Database.release_conn(conn)
      return clients
   except Exception as e:
      print(f"Error fetching clients: {e}")
      return []


def add_transaction(description, amount, category, client_id):
   try:
      conn = Database.get_conn()
      with conn.cursor() as c:
         c.execute(
            "INSERT INTO transactions (description, amount, category, date, client_id) VALUES (%s, %s, %s, CURRENT_DATE, %s)",
            (description, amount, category, client_id))
         conn.commit()
      Database.release_conn(conn)
   except Exception as e:
      print(f"Error adding transaction: {e}")


def get_transactions(client_id=None):
   try:
      conn = Database.get_conn()
      with conn.cursor() as c:
         if client_id:
            c.execute("SELECT id, description, amount, category, date FROM transactions WHERE client_id = %s",
                      (client_id,))
         else:
            c.execute("SELECT id, description, amount, category, date FROM transactions")
         transactions = c.fetchall()
      Database.release_conn(conn)
      return transactions
   except Exception as e:
      print(f"Error fetching transactions: {e}")
      return []# models/generative_report.py
import pandas as pd
from datetime import datetime
import markdown

def generate_report(df):
    try:
        total_income = df[df["Category"] == "income"]["Amount"].sum()
        total_expenses = df[df["Category"] == "expense"]["Amount"].sum()
        net_profit = total_income - total_expenses

        report = f"""
# Financial Report - {datetime.now().strftime('%Y-%m-%d')}

## Overview
- **Total Income**: ₹{total_income:.2f}
- **Total Expenses**: ₹{total_expenses:.2f}
- **Net Profit**: ₹{net_profit:.2f}

## Insights
- **Top Expense Category**: {df[df["Category"] == "expense"]["Description"].mode()[0] if not df[df["Category"] == "expense"].empty else "N/A"}
- **Largest Income Source**: {df[df["Category"] == "income"]["Description"].mode()[0] if not df[df["Category"] == "income"].empty else "N/A"}

## Recommendations
- Optimize high-expense categories.
- Explore additional income streams.
"""
        return markdown.markdown(report)
    except Exception as e:
        print(f"Error in generate_report: {e}")
        return "<p>Error generating report.</p>"