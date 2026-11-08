import psycopg2
import os

windows_host_ip = os.popen(
    "ip route | grep default | awk '{print $3}'"
).read().strip()

DB_CONFIG = {
    "host": windows_host_ip,
    "port": 5432,
    "database": "guce_siat_cc",
    "user": "postgres",
    "password": "nobodycan@2H"
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)