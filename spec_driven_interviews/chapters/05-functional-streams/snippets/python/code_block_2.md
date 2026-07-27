```python
# Imperative anti-pattern: Hard to read, mutable state, difficult to parallelize
volumes = {}
for tx in transactions:
    if tx.amount >= threshold:
        merchant_id = tx.destination_account_id
        volumes[merchant_id] = volumes.get(merchant_id, 0) + tx.amount
```
