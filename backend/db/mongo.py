# Mongo DB client(reusable)
# 1 client per application, created at startup of fastapi and shutdown at closed, stored in app state.

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from typing import Optional


class MongoDB:
    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db = None

    def connect(self) -> None:
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            print("[MongoDB] Connected successfully")
        except PyMongoError as e:
            self.client = None
            self.db = None
            print(f"[MongoDB] Connection failed: {e}")

    def close(self) -> None:
        if self.client:
            self.client.close()
            print("[MongoDB] Connection closed")
