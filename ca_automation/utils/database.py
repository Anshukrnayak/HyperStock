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
            c.execute("INSERT INTO transactions (description, amount, category, date, client_id) VALUES (%s, %s, %s, CURRENT_DATE, %s)",
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
                c.execute("SELECT id, description, amount, category, date FROM transactions WHERE client_id = %s", (client_id,))
            else:
                c.execute("SELECT id, description, amount, category, date FROM transactions")
            transactions = c.fetchall()
        Database.release_conn(conn)
        return transactions
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        return []