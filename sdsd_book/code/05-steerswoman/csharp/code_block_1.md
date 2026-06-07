```csharp
// Steerswoman Interface
public interface ITransactionRepository
{
    Task<IEnumerable<Transaction>> FetchTransactionsAsync(string userId, int limit, int offset);
}
```