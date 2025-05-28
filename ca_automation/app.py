# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from datetime import datetime, timedelta
from models.ml_classifier import predict_category, active_learning_query
from models.lstm_forecast import forecast_expenses
from models.nlp_assistant import answer_query
from models.ocr_parser import parse_invoice
from models.rl_portfolio import optimize_portfolio_rl
from models.fraud_detector import detect_fraud
from models.generative_report import generate_report
from utils.database import init_db, add_transaction, get_transactions, add_client, get_clients
from utils.tax_calculator import calculate_gst, calculate_tds
from utils.receivables import aging_analysis, send_reminder
from utils.blockchain import log_transaction
from utils.portfolio import black_scholes_price

# Load configuration
with open("config.yaml", "r") as f:
   config = yaml.safe_load(f)

# Initialize database
init_db()

# Streamlit configuration
st.set_page_config(page_title="AI CA Beast", layout="wide")
st.markdown('<style>body {font-family: Arial, sans-serif;}</style>', unsafe_allow_html=True)
st.markdown('<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">',
            unsafe_allow_html=True)

# Session state
if 'role' not in st.session_state:
   st.session_state.role = 'accountant'
if 'client_id' not in st.session_state:
   st.session_state.client_id = None

# Sidebar navigation
st.sidebar.title("AI CA Beast")
st.sidebar.markdown('<div class="bg-gray-900 text-white p-4 rounded-lg">Financial Ecosystem</div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Navigate",
                        ["Dashboard", "Add Transaction", "Portfolio", "Virtual CA", "Taxation", "OCR Upload",
                         "Receivables", "CA Dashboard", "WhatsApp Invoice", "Fraud Detection", "Reports"])

# Client selection
if st.session_state.role == 'accountant':
   clients = get_clients()
   client_options = {client[1]: client[0] for client in clients}
   selected_client = st.sidebar.selectbox("Select Client", ["All"] + list(client_options.keys()))
   st.session_state.client_id = client_options.get(selected_client) if selected_client != "All" else None

# Dashboard
if page == "Dashboard":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Financial Ecosystem Dashboard</h1>',
               unsafe_allow_html=True)

   transactions = get_transactions(st.session_state.client_id)
   df = pd.DataFrame(transactions, columns=["ID", "Description", "Amount", "Category", "Date"])

   total_income = df[df["Category"] == "income"]["Amount"].sum()
   total_expenses = df[df["Category"] == "expense"]["Amount"].sum()
   net_profit = total_income - total_expenses

   col1, col2, col3 = st.columns(3)
   with col1:
      st.markdown(
         '<div class="bg-white p-6 rounded-lg shadow-xl"><h2 class="text-xl font-semibold text-gray-700">Total Income</h2><p class="text-3xl text-green-600">₹{:.2f}</p></div>'.format(
            total_income), unsafe_allow_html=True)
   with col2:
      st.markdown(
         '<div class="bg-white p-6 rounded-lg shadow-xl"><h2 class="text-xl font-semibold text-gray-700">Total Expenses</h2><p class="text-3xl text-red-600">₹{:.2f}</p></div>'.format(
            total_expenses), unsafe_allow_html=True)
   with col3:
      st.markdown(
         '<div class="bg-white p-6 rounded-lg shadow-xl"><h2 class="text-xl font-semibold text-gray-700">Net Profit</h2><p class="text-3xl text-blue-600">₹{:.2f}</p></div>'.format(
            net_profit), unsafe_allow_html=True)

   st.markdown('<h2 class="text-2xl font-semibold mt-8 text-gray-700">Transaction Trends</h2>', unsafe_allow_html=True)
   if not df.empty:
      df["Date"] = pd.to_datetime(df["Date"])
      trend = df.groupby("Date")["Amount"].sum().reset_index()
      fig = px.line(trend, x="Date", y="Amount", title="Transaction Trends")
      st.plotly_chart(fig)

   if len(df) > 10:
      forecast = forecast_expenses(df)
      if not forecast.empty:
         st.markdown('<h2 class="text-2xl font-semibold mt-8 text-gray-700">AI Expense Forecast</h2>',
                     unsafe_allow_html=True)
         fig = px.line(forecast, x="Date", y="Amount", title="Expense Forecast")
         st.plotly_chart(fig)

# Add Transaction
elif page == "Add Transaction":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Add Transaction</h1>',
               unsafe_allow_html=True)
   with st.form("transaction_form"):
      description = st.text_input("Description", placeholder="e.g., Office Supplies")
      amount = st.number_input("Amount", min_value=0.0, step=0.01)
      submitted = st.form_submit_button("Add Transaction", use_container_width=True)
      if submitted:
         if description and amount > 0:
            category, confidence = predict_category(description)
            add_transaction(description, amount, category, st.session_state.client_id)
            log_transaction(description, amount, category)
            if confidence < 0.7:
               active_learning_query(description, category)
            st.success(f"Transaction added! Category: {category} (Confidence: {confidence:.2f})")
         else:
            st.error("Please provide a valid description and amount.")

# Portfolio Optimization
elif page == "Portfolio":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Portfolio Optimization</h1>',
               unsafe_allow_html=True)
   weights = optimize_portfolio_rl()
   st.markdown('<h2 class="text-2xl font-semibold text-gray-700">RL-Optimized Portfolio Weights</h2>',
               unsafe_allow_html=True)
   for i, weight in enumerate(weights, 1):
      st.markdown(f'<p class="text-lg text-gray-600">Asset {i}: {weight:.4f}</p>', unsafe_allow_html=True)

   st.markdown('<h2 class="text-2xl font-semibold mt-8 text-gray-700">Options Pricing (Black-Scholes)</h2>',
               unsafe_allow_html=True)
   option_price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
   st.markdown(f'<p class="text-lg text-gray-600">Call Option Price: ₹{option_price:.2f}</p>', unsafe_allow_html=True)

# Virtual CA
elif page == "Virtual CA":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Conversational CA Assistant</h1>',
               unsafe_allow_html=True)
   query = st.text_area("Ask your Virtual CA", placeholder="e.g., What are my top expenses? How can I optimize taxes?")
   if query:
      response = answer_query(query, get_transactions(st.session_state.client_id))
      st.markdown(
         f'<div class="bg-white p-6 rounded-lg shadow-xl"><p class="text-lg text-gray-600">{response}</p></div>',
         unsafe_allow_html=True)

# Taxation
elif page == "Taxation":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Taxation & Compliance</h1>',
               unsafe_allow_html=True)
   transactions = get_transactions(st.session_state.client_id)
   df = pd.DataFrame(transactions, columns=["ID", "Description", "Amount", "Category", "Date"])

   if not df.empty:
      gst = calculate_gst(df)
      tds = calculate_tds(df)
      st.markdown(
         f'<div class="bg-white p-6 rounded-lg shadow-xl"><h2 class="text-xl font-semibold text-gray-700">GST Payable</h2><p class="text-3xl text-blue-600">₹{gst:.2f}</p></div>',
         unsafe_allow_html=True)
      st.markdown(
         f'<div class="bg-white p-6 rounded-lg shadow-xl"><h2 class="text-xl font-semibold text-gray-700">TDS Deducted</h2><p class="text-3xl text-blue-600">₹{tds:.2f}</p></div>',
         unsafe_allow_html=True)
   else:
      st.info("No transactions available for tax calculation.")

# OCR Upload
elif page == "OCR Upload":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Upload Invoice</h1>',
               unsafe_allow_html=True)
   uploaded_file = st.file_uploader("Upload Invoice Image", type=["png", "jpg", "jpeg"])
   if uploaded_file:
      description, amount = parse_invoice(uploaded_file)
      if description and amount:
         category, confidence = predict_category(description)
         add_transaction(description, amount, category, st.session_state.client_id)
         log_transaction(description, amount, category)
         st.success(f"Invoice parsed! Description: {description}, Amount: ₹{amount}, Category: {category}")
      else:
         st.error("Could not parse invoice. Please try another image.")

# Receivables
elif page == "Receivables":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Receivables Analysis</h1>',
               unsafe_allow_html=True)
   transactions = get_transactions(st.session_state.client_id)
   df = pd.DataFrame(transactions, columns=["ID", "Description", "Amount", "Category", "Date"])
   aging_df = aging_analysis(df)

   if not aging_df.empty:
      st.markdown('<h2 class="text-2xl font-semibold mt-8 text-gray-700">Aging Analysis</h2>', unsafe_allow_html=True)
      st.dataframe(aging_df)
      if st.button("Send Payment Reminders", use_container_width=True):
         reminders = send_reminder(aging_df)
         for reminder in reminders:
            st.markdown(
               f'<div class="bg-white p-6 rounded-lg shadow-xl"><p class="text-lg text-gray-600">{reminder}</p></div>',
               unsafe_allow_html=True)
   else:
      st.info("No receivables data available.")

# CA Dashboard
elif page == "CA Dashboard":
   if st.session_state.role != 'accountant':
      st.error("Access restricted to accountants.")
   else:
      st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">CA Multi-Client Dashboard</h1>',
                  unsafe_allow_html=True)
      with st.form("client_form"):
         client_name = st.text_input("Add New Client", placeholder="e.g., ABC Corp")
         submitted = st.form_submit_button("Add Client", use_container_width=True)
         if submitted and client_name:
            add_client(client_name)
            st.success(f"Client {client_name} added!")

      clients = get_clients()
      for client_id, client_name in clients:
         st.markdown(f'<h2 class="text-2xl font-semibold mt-8 text-gray-700">{client_name}</h2>',
                     unsafe_allow_html=True)
         transactions = get_transactions(client_id)
         df = pd.DataFrame(transactions, columns=["ID", "Description", "Amount", "Category", "Date"])
         st.dataframe(df[["Description", "Amount", "Category", "Date"]])

# WhatsApp Invoice
elif page == "WhatsApp Invoice":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Generate Invoice (WhatsApp)</h1>',
               unsafe_allow_html=True)
   with st.form("invoice_form"):
      description = st.text_input("Invoice Description", placeholder="e.g., Consulting Services")
      amount = st.number_input("Invoice Amount", min_value=0.0, step=0.01)
      client_phone = st.text_input("Client Phone (Simulated)", placeholder="e.g., +91 9876543210")
      submitted = st.form_submit_button("Generate Invoice", use_container_width=True)
      if submitted and description and amount > 0:
         invoice = f"Invoice: {description}, Amount: ₹{amount}, Sent to: {client_phone}"
         add_transaction(description, amount, "income", st.session_state.client_id)
         log_transaction(description, amount, "income")
         st.success(f"Simulated WhatsApp invoice: {invoice}")
      elif submitted:
         st.error("Please provide a valid description and amount.")

# Fraud Detection
elif page == "Fraud Detection":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Fraud Detection</h1>',
               unsafe_allow_html=True)
   transactions = get_transactions(st.session_state.client_id)
   df = pd.DataFrame(transactions, columns=["ID", "Description", "Amount", "Category", "Date"])
   if not df.empty:
      anomalies = detect_fraud(df)
      if not anomalies.empty:
         st.markdown('<h2 class="text-2xl font-semibold mt-8 text-gray-700">Suspicious Transactions</h2>',
                     unsafe_allow_html=True)
         st.dataframe(anomalies)
      else:
         st.info("No suspicious transactions detected.")
   else:
      st.info("No transactions available for fraud detection.")

# Reports
elif page == "Reports":
   st.markdown('<h1 class="text-4xl font-bold text-center mb-8 text-gray-800">AI-Generated Reports</h1>',
               unsafe_allow_html=True)
   transactions = get_transactions(st.session_state.client_id)
   df = pd.DataFrame(transactions, columns=["ID", "Description", "Amount", "Category", "Date"])
   if not df.empty:
      report = generate_report(df)
      st.markdown(report, unsafe_allow_html=True)
   else:
      st.info("No transactions available for reporting.")