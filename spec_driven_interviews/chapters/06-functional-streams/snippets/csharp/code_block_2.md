```csharp
// Imperative anti-pattern: Hard to read, mutable state, difficult to parallelize
var volumes = new Dictionary<Guid, decimal>();
foreach (var tx in transactions) {
    if (tx.Amount >= threshold) {
        var merchantId = tx.DestinationAccountId;
        if (!volumes.TryGetValue(merchantId, out decimal currentSum)) {
            currentSum = 0;
        }
        volumes[merchantId] = currentSum + tx.Amount;
    }
}
```
