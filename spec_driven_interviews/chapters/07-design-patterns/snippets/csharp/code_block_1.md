```csharp
using System;
using System.Collections.Generic;

namespace AuraPay.Events
{
    /// <summary>
    /// Interface defining the Observer contract for transaction events.
    /// </summary>
    public interface ITransactionObserver
    {
        void OnTransactionSuccess(TransactionRecord transaction);
        void OnTransactionFailed(TransactionRecord transaction, Exception error);
    }

    /// <summary>
    /// Concrete Observer that writes a persistent audit trail for security compliance.
    /// </summary>
    public class AuditTrailObserver : ITransactionObserver
    {
        public void OnTransactionSuccess(TransactionRecord transaction)
        {
            Console.WriteLine($"AUDIT SUCCESS: Transaction {transaction.TransactionId} of {transaction.Amount} " +
                              $"{transaction.Currency} from {transaction.SourceAccountId} to {transaction.DestinationAccountId} " +
                              $"registered in immutable log.");
        }

        public void OnTransactionFailed(TransactionRecord transaction, Exception error)
        {
            Console.Error.WriteLine($"AUDIT FAILURE: Transaction {transaction.TransactionId} failed. Error: {error.Message}");
        }
    }

    /// <summary>
    /// Subject class managing observers and publishing transaction status updates.
    /// </summary>
    public class TransactionEventPublisher
    {
        private readonly List<ITransactionObserver> _observers = new List<ITransactionObserver>();
        private readonly object _lock = new object();

        public void RegisterObserver(ITransactionObserver observer)
        {
            if (observer == null) throw new ArgumentNullException(nameof(observer));
            lock (_lock)
            {
                _observers.Add(observer);
            }
        }

        public void DeregisterObserver(ITransactionObserver observer)
        {
            lock (_lock)
            {
                _observers.Remove(observer);
            }
        }

        public void NotifySuccess(TransactionRecord transaction)
        {
            List<ITransactionObserver> targets;
            lock (_lock)
            {
                targets = new List<ITransactionObserver>(_observers);
            }
            foreach (var observer in targets)
            {
                observer.OnTransactionSuccess(transaction);
            }
        }

        public void NotifyFailure(TransactionRecord transaction, Exception error)
        {
            List<ITransactionObserver> targets;
            lock (_lock)
            {
                targets = new List<ITransactionObserver>(_observers);
            }
            foreach (var observer in targets)
            {
                observer.OnTransactionFailed(transaction, error);
            }
        }
    }
}
```
