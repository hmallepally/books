```python
def log_and_map(t):
    log.debug(f"Passed Filter: {t.id}")
    return t.merchant_id

merchant_ids = [log_and_map(t) for t in transactions if t.amount > 100]
```