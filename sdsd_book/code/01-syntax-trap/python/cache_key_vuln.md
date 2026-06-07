```python
def generate_cache_key(request: Request) -> str:
    query_string = request.url.query
    key_hash = hashlib.md5(query_string.encode()).hexdigest()
    return f"cache:transactions:{key_hash}"
```