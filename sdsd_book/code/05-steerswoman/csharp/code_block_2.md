```csharp
// Steerswoman Concrete
public class TransactionRepository : ITransactionRepository
{
    private readonly AppDbContext _db;
    public TransactionRepository(AppDbContext db) => _db = db;

    public async Task<IEnumerable<Transaction>> FetchTransactionsAsync(string userId, int limit, int offset)
    {
        return await _db.Transactions
            .Where(t => t.UserId == userId)
            .Skip(offset)
            .Take(limit)
            .ToListAsync();
    }
}
```