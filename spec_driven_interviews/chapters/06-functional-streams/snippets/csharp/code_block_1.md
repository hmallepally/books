```csharp
using System;
using System.Collections.Generic;
using System.Linq;

namespace AuraPay.Analytics
{
    /// <summary>
    /// Demonstrates high-performance batch transaction analytics using C# LINQ.
    /// </summary>
    public class TransactionAnalytics
    {
        /// <summary>
        /// Processes a list of transactions to aggregate total volume per merchant,
        /// filtering out low-value records.
        /// </summary>
        public Dictionary<Guid, decimal> AggregateMerchantVolumes(
            List<TransactionRecord> transactions, 
            decimal minAmountThreshold)
        {
            if (transactions == null) throw new ArgumentNullException(nameof(transactions));

            // Declarative LINQ query syntax
            return transactions
                .Where(t => t.Amount >= minAmountThreshold)
                .GroupBy(t => t.DestinationAccountId)
                .ToDictionary(
                    group => group.Key,
                    group => group.Sum(t => t.Amount)
                );
        }

        /// <summary>
        /// Finds the transaction IDs of all transfers exceeding a safety limit.
        /// </summary>
        public List<Guid> GetHighValueTransactionIds(List<TransactionRecord> transactions, decimal limit)
        {
            return transactions
                .Where(t => t.Amount > limit)
                .Select(t => t.TransactionId)
                .ToList();
        }
    }
}
```
