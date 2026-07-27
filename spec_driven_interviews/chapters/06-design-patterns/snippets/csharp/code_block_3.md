```csharp
// Wrapping the core processor with an audit logging decorator
ITransactionProcessor decoratedProcessor = new AuditingTransactionProcessorDecorator(
    new CoreTransactionProcessor(repository, calculator, sender)
);
```
