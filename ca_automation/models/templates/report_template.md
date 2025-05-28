# Financial Report - {{ date }}

## Overview
- **Total Income**: ₹{{ total_income }}
- **Total Expenses**: ₹{{ total_expenses }}
- **Net Profit**: ₹{{ net_profit }}

## Insights
- **Top Expense Category**: {{ top_expense }}
- **Largest Income Source**: {{ top_income }}

## Recommendations
{{ recommendations }}

# config.yaml
database:
  host: localhost
  port: 5432
  user: postgres
  password: password
  name: ca_automation
blockchain:
  provider: http://127.0.0.1:8545
  contract_address: 0xYourContractAddress
api:
  gst: https://api.gst.gov.in
models:
  nlp: distilbert-base-cased-distilled-squad
