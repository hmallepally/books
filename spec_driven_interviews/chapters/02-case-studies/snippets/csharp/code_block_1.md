```csharp
using System;

namespace AuraPay.Domain
{
    /// <summary>
    /// Represents an immutable, validated financial transaction record in AuraPay.
    /// Enforces pre-conditions on initialization.
    /// </summary>
    public record TransactionRecord
    {
        public Guid TransactionId { get; init; }
        public Guid SourceAccountId { get; init; }
        public Guid DestinationAccountId { get; init; }
        public decimal Amount { get; init; }
        public string Currency { get; init; }
        public DateTime Timestamp { get; init; }

        public TransactionRecord(
            Guid transactionId,
            Guid sourceAccountId,
            Guid destinationAccountId,
            decimal amount,
            string currency,
            DateTime timestamp)
        {
            if (transactionId == Guid.Empty) throw new ArgumentException("Transaction ID cannot be empty", nameof(transactionId));
            if (sourceAccountId == Guid.Empty) throw new ArgumentException("Source Account ID cannot be empty", nameof(sourceAccountId));
            if (destinationAccountId == Guid.Empty) throw new ArgumentException("Destination Account ID cannot be empty", nameof(destinationAccountId));
            if (string.IsNullOrWhiteSpace(currency)) throw new ArgumentException("Currency code cannot be empty", nameof(currency));
            if (amount <= 0) throw new ArgumentException("Transaction amount must be strictly positive", nameof(amount));
            if (sourceAccountId == destinationAccountId) throw new ArgumentException("Source and destination accounts must be distinct");

            TransactionId = transactionId;
            SourceAccountId = sourceAccountId;
            DestinationAccountId = destinationAccountId;
            Amount = amount;
            Currency = currency;
            Timestamp = timestamp;
        }
    }
}
```
