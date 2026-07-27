```csharp
var merchantIds = transactions
    .Where(t => t.Amount > 100)
    .Select(t => {
        log.Debug($"Passed Filter: {t.Id}");
        return t.MerchantId;
    })
    .ToList();
```