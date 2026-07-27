```python
# Example of a fluent, type-safe builder for transactions
tx = (TransactionRecordBuilder()
    .with_id(uuid.uuid4())
    .from_account(source_id)
    .to_account(dest_id)
    .with_amount(Decimal("100.00"))
    .in_currency("USD")
    .at_timestamp(datetime.now(timezone.utc))
    .build()) # Immutability and invariants are validated in build()
```
