```python
def generate_cache_key(request: Request, user_id: str) -> str:
    """Generates a tenant-isolated cache key."""
    query_string = request.url.query
    key_hash = hashlib.md5(query_string.encode()).hexdigest()
    # The cache key is now rigidly bound to the tenant ID
    return f"cache:tenant:{user_id}:transactions:{key_hash}"
```