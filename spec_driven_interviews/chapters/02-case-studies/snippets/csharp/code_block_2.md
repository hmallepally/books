```csharp
using System;

namespace AuraPay.Domain
{
    /// <summary>
    /// Represents a stateful Ledger Account in AuraPay, enforcing business invariants
    /// during state transitions.
    /// </summary>
    public class LedgerAccount
    {
        private readonly object _lock = new object();
        public Guid AccountId { get; }
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

        public LedgerAccount(Guid accountId, string currency, decimal initialBalance, decimal overdraftLimit)
        {
            if (accountId == Guid.Empty) throw new ArgumentException("Account ID cannot be empty", nameof(accountId));
            if (string.IsNullOrWhiteSpace(currency)) throw new ArgumentException("Currency code cannot be empty", nameof(currency));
            if (overdraftLimit < 0) throw new ArgumentException("Overdraft limit cannot be negative", nameof(overdraftLimit));
            if (initialBalance + overdraftLimit < 0) throw new ArgumentException("Initial balance violates the overdraft limit");

            AccountId = accountId;
            Currency = currency;
            _balance = initialBalance;
            OverdraftLimit = overdraftLimit;
        }

        /// <summary>
        /// Credits the account. Enforces positive credit amount.
        /// </summary>
        public void Credit(decimal amount)
        {
            if (amount <= 0) throw new ArgumentException("Credit amount must be positive", nameof(amount));
            lock (_lock)
            {
                _balance += amount;
            }
        }

        /// <summary>
        /// Debits the account. Enforces balance invariants and overdraft limits.
        /// </summary>
        public void Debit(decimal amount)
        {
            if (amount <= 0) throw new ArgumentException("Debit amount must be positive", nameof(amount));
            lock (_lock)
            {
                decimal newBalance = _balance - amount;
                // INVARIANT ENFORCEMENT
                if (newBalance + OverdraftLimit < 0)
                {
                    throw new InvalidOperationException(
                        $"Debit of {amount} exceeds account overdraft boundary. " +
                        $"Balance: {_balance}, Limit: -{OverdraftLimit}");
                }
                _balance = newBalance;
            }
        }
    }
}
```
