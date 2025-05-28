from web3 import Web3
import json

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
contract_address = "0xYourContractAddress"
with open("contract_abi.json", "r") as f:
    contract_abi = json.load(f)
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def log_transaction(description, amount, category):
    try:
        tx = contract.functions.logTransaction(description, float(amount), category).buildTransaction({
            'from': w3.eth.accounts[0],
            'nonce': w3.eth.getTransactionCount(w3.eth.accounts[0]),
            'gas': 2000000,
            'gasPrice': w3.toWei('20', 'gwei')
        })
        signed_tx = w3.eth.account.signTransaction(tx, private_key="your_private_key")
        tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return tx_hash.hex()
    except Exception as e:
        print(f"Error in log_transaction: {e}")
        return None