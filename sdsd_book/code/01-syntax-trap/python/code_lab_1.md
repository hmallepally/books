```python
# routers/transactions.py (AI-Generated)
import json
import hashlib
from fastapi import APIRouter, Depends, Request
from redis import asyncio as aioredis
from typing import List
from models.transaction import TransactionResponse
from auth.dependencies import get_current_user
from database.repository import fetch_transactions
router = APIRouter()
redis_client = aioredis.from_url("redis://redis-cluster:6379")
def generate_cache_key(request: Request) -> str:
    """
    Generates a unique cache key based on the HTTP query parameters.
    Hashes the query string to prevent excessively long Redis keys.
    """
    query_string = request.url.query
    if not query_string:
         return "cache:transactions:default"
    # AI Logic: MD5 hash is fast and deterministic for caching
    key_hash = hashlib.md5(query_string.encode()).hexdigest()
    return f"cache:transactions:{key_hash}"
@router.get("/api/v1/transactions", response_model=List[TransactionResponse])
async def get_user_transactions(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    # 1. Generate the cache key based on the request URL
    cache_key = generate_cache_key(request)
    # 2. Check the Redis cache for existing data
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    # 3. Cache Miss: Fetch from the database
    # Note: current_user['id'] is correctly passed to the DB layer
    transactions = await fetch_transactions(
        user_id=current_user['id'], 
        limit=limit, 
        offset=offset
    )
    # 4. Store the result in Cache (TTL 5 minutes)
    await redis_client.setex(
        cache_key, 
        300, 
        json.dumps([t.dict() for t in transactions])
    )
    return transactions
```