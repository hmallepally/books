```java
// Wrapping the core processor with an audit logging decorator
TransactionProcessor decoratedProcessor = new AuditingTransactionProcessorDecorator(
    new CoreTransactionProcessor(repository, calculator, sender)
);
```
