```python
# Wrapping the core processor with an audit logging decorator
decorated_processor = AuditingTransactionProcessorDecorator(
    CoreTransactionProcessor(repository, calculator, sender)
)
```
