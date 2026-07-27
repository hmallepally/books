```csharp
using System;

namespace AuraPay.Domain
{
    /// <summary>
    /// Demonstrates a rich domain model encapsulating transfer logic and enforcing 
    /// cross-entity invariants.
    /// </summary>
    public class LedgerAccount
    {
        private readonly object _lock = new object();
        public string AccountId { get; }
        public string Currency { get; }
        private decimal _balance;
        public decimal OverdraftLimit { get; }

        public decimal Balance
        {
            get
            {
                lock (_lock)
                {
                    return _balance;
                }
            }
        }

        public LedgerAccount(string accountId, string currency, decimal initialBalance, decimal overdraftLimit)
        {
            AccountId = accountId ?? throw new ArgumentNullException(nameof(accountId));
            Currency = currency ?? throw new ArgumentNullException(nameof(currency));
            _balance = initialBalance;
            OverdraftLimit = overdraftLimit;
        }

        public void Debit(decimal amount)
        {
            if (amount <= 0) throw new ArgumentException("Debit amount must be positive", nameof(amount));
            lock (_lock)
            {
                decimal newBalance = _balance - amount;
                if (newBalance + OverdraftLimit < 0)
                {
                    throw new InvalidOperationException("Overdraft limit exceeded");
                }
                _balance = newBalance;
            }
        }

        public void Credit(decimal amount)
        {
            if (amount <= 0) throw new ArgumentException("Credit amount must be positive", nameof(amount));
            lock (_lock)
            {
                _balance += amount;
            }
        }

        /// <summary>
        /// Executes a thread-safe transfer to a target account, enforcing business invariants.
        /// Prevents mismatched currencies and double-debiting.
        /// </summary>
        public void TransferTo(LedgerAccount target, decimal amount)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));

            // PRE-CONDITION ENFORCEMENT: Currency matching
            if (Currency != target.Currency)
            {
                throw new InvalidOperationException($"Cannot transfer between mismatched currencies: {Currency} and {target.Currency}");
            }

            // PRE-CONDITION ENFORCEMENT: Self-transfer check
            if (AccountId == target.AccountId)
            {
                throw new ArgumentException("Cannot transfer to the same account");
            }

            // To prevent deadlocks, lock accounts in a stable global order
            var firstLock = string.Compare(AccountId, target.AccountId, StringComparison.Ordinal) < 0 ? this : target;
            var secondLock = firstLock == this ? target : this;

            lock (firstLock._lock)
            {
                lock (secondLock._lock)
                {
                    // Execute atomic debit-credit sequence
                    this.Debit(amount);
                    target.Credit(amount);
                }
            }
        }
    }
}
```
