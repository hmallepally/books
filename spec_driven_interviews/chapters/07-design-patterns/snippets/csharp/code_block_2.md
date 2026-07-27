```csharp
// Example of a fluent, type-safe builder for transactions
TransactionRecord tx = new TransactionRecordBuilder()
    .WithId(Guid.NewGuid())
    .FromAccount(sourceId)
    .ToAccount(destId)
    .WithAmount(100.00m)
    .InCurrency("USD")
    .AtTimestamp(DateTimeOffset.UtcNow)
    .Build(); // Immutability and invariants are validated in Build()
```
