from settings import *
from pymongo import MongoClient

def drop_all_db():
    db_url = os.getenv('DB_URL')
    db_name = os.getenv('DB_NAME')
    client = MongoClient(db_url)
    db = client[db_name]
    db['persons'].drop()
    db['events'].drop()

if __name__ == '__main__':
    drop_all_db()