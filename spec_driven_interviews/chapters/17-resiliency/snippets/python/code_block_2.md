```python
# Anti-pattern: Dual-Write
def complete_transaction(tx: TransactionRecord) -> None:
    database.save(tx) # Database Write
    kafka_producer.send("transaction-topic", tx) # Network Call
```
