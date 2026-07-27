```csharp
// Anemic Account Model (Fragile Data Holder)
public class Account 
{
    public string Id { get; set; }
    public decimal Balance { get; set; }
    public string Currency { get; set; }
}

// Stateless Service containing business invariants (Anti-pattern)
public class LedgerService 
{
    public void Transfer(Account from, Account to, decimal amount) 
    {
        if (from.Balance < amount) 
        {
            throw new ArgumentException("Insufficient funds");
        }
        if (from.Currency != to.Currency) 
        {
            throw new ArgumentException("Currency mismatch");
        }
        from.Balance -= amount;
        to.Balance += amount;
    }
}
```