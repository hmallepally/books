```python
from abc import ABC, abstractmethod
from decimal import Decimal
from uuid import UUID

class LedgerRepository(ABC):
    """
    Abstraction for database operations (Dependency Inversion Principle).
    """
    @abstractmethod
    def find_by_id(self, account_id: UUID):
        pass

    @abstractmethod
    def save(self, account):
        pass

class FeeCalculator(ABC):
    """
    Abstraction for fee calculations (Open/Closed Principle).
    """
    @abstractmethod
    def calculate(self, transaction) -> Decimal:
        pass

class TransactionNotificationSender(ABC):
    """
    Interface Segregation Principle: Focused notification dispatch interface.
    """
    @abstractmethod
    def send_notification(self, transaction, status: str):
        pass

class TransactionProcessor:
    """
    Core transaction processor showing SOLID compliance.
    """
    def __init__(self, repository: LedgerRepository, fee_calculator: FeeCalculator, notification_sender: TransactionNotificationSender):
        self.repository = repository
        self.fee_calculator = fee_calculator
        self.notification_sender = notification_sender

    def process(self, transaction):
        if not transaction:
            raise ValueError("Transaction cannot be null")

        # 1. Retrieve accounts from abstraction (DIP)
        source = self.repository.find_by_id(transaction.source_account_id)
        destination = self.repository.find_by_id(transaction.destination_account_id)

        if not source or not destination:
            raise ValueError("Source or destination account not found")

        # 2. Calculate fee dynamically (OCP)
        fee = self.fee_calculator.calculate(transaction)
        total_debit = transaction.amount + fee

        # 3. Coordinate state transitions on rich domain objects (SRP / LSP)
        source.debit(total_debit)
        destination.credit(transaction.amount)

        # 4. Persist updated states (DIP)
        self.repository.save(source)
        self.repository.save(destination)

        # 5. Notify via segregated interface (ISP)
        self.notification_sender.send_notification(transaction, "SUCCESS")
```
