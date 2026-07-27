```python
from abc import ABC, abstractmethod

class TransactionObserver(ABC):
    """
    Interface defining the Observer contract for transaction events.
    """
    @abstractmethod
    def on_transaction_success(self, transaction):
        pass

    @abstractmethod
    def on_transaction_failed(self, transaction, error: Exception):
        pass

class AuditTrailObserver(TransactionObserver):
    """
    Concrete Observer that writes a persistent audit trail for security compliance.
    """
    def on_transaction_success(self, transaction):
        print(f"AUDIT SUCCESS: Transaction {transaction.transaction_id} of {transaction.amount} "
              f"{transaction.currency} from {transaction.source_account_id} to {transaction.destination_account_id} "
              f"registered in immutable log.")

    def on_transaction_failed(self, transaction, error: Exception):
        print(f"AUDIT FAILURE: Transaction {transaction.transaction_id} failed. Error: {str(error)}")

class TransactionEventPublisher:
    """
    Subject class managing observers and publishing transaction status updates.
    """
    def __init__(self):
        self._observers = []

    def register_observer(self, observer: TransactionObserver):
        self._observers.append(observer)

    def deregister_observer(self, observer: TransactionObserver):
        self._observers.remove(observer)

    def notify_success(self, transaction):
        for observer in self._observers:
            observer.on_transaction_success(transaction)

    def notify_failure(self, transaction, error: Exception):
        for observer in self._observers:
            observer.on_transaction_failed(transaction, error)
```
