```csharp
// Anti-pattern: Dual-Write
public void CompleteTransaction(TransactionRecord tx) {
    _database.Save(tx); // Database Write
    _kafkaTemplate.Send("transaction-topic", tx); // Network Call
}
```
