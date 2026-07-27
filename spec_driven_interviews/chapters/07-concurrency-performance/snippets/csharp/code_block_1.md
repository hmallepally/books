```csharp
using System;

namespace AuraPay.Persistence
{
    /// <summary>
    /// Represents a database-mapped Ledger Account Entity with versioning for
    /// Optimistic Concurrency Control (OCC).
    /// </summary>
    public class AccountEntity
    {
        public Guid Id { get; }
        public decimal Balance { get; private set; }
        public string Currency { get; }
        public long Version { get; private set; }

        public AccountEntity(Guid id, decimal balance, string currency, long version)
        {
            Id = id;
            Balance = balance;
            Currency = currency ?? throw new ArgumentNullException(nameof(currency));
            Version = version;
        }

        public void UpdateBalance(decimal newBalance)
        {
            Balance = newBalance;
        }

        public void IncrementVersion()
        {
            Version++;
        }
    }

    /// <summary>
    /// Repository implementation executing the version check update query.
    /// </summary>
    public class DatabaseLedgerRepository
    {
        /// <summary>
        /// Updates the account in the database using a strict version-matching query.
        /// Throws an exception if another thread modified the record concurrently.
        /// </summary>
        public void Save(AccountEntity account)
        {
            if (account == null) throw new ArgumentNullException(nameof(account));

            // Simulates SQL database update query:
            // UPDATE accounts SET balance = @balance, version = version + 1 WHERE id = @id AND version = @version;
            string query = "UPDATE accounts SET balance = @Balance, version = @Version + 1 WHERE id = @Id AND version = @Version";

            int rowsUpdated = MockExecuteUpdateQuery(query, account);

            // OCC FAILURE CHECK: If rowsUpdated is 0, a concurrent thread modified this record first.
            if (rowsUpdated == 0)
            {
                throw new InvalidOperationException(
                    $"Optimistic lock conflict on account {account.Id}. Outdated version: {account.Version}"
                );
            }

            account.IncrementVersion();
        }

        private int MockExecuteUpdateQuery(string query, AccountEntity account)
        {
            // Simulates database execution
            return 1; // 1 means success; 0 means no record matched (concurrency mismatch)
        }
    }
}
```
