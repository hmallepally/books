```python
from decimal import Decimal
import threading

class LedgerAccount:
    """
    Demonstrates a rich domain model encapsulating transfer logic and enforcing 
    cross-entity invariants.
    """
    def __init__(self, account_id: str, currency: str, initial_balance: Decimal, overdraft_limit: Decimal):
        self.account_id = account_id
        self.currency = currency
        self._balance = initial_balance
        self.overdraft_limit = overdraft_limit
        self._lock = threading.RLock()

    @property
    def balance(self) -> Decimal:
        with self._lock:
            return self._balance

    def debit(self, amount: Decimal):
        if amount <= 0:
            raise ValueError("Debit amount must be positive")
        with self._lock:
            new_balance = self._balance - amount
            if new_balance + self.overdraft_limit < 0:
                raise ValueError("Overdraft limit exceeded")
            self._balance = new_balance

    def credit(self, amount: Decimal):
        if amount <= 0:
            raise ValueError("Credit amount must be positive")
        with self._lock:
            self._balance += amount

    def transfer_to(self, target: 'LedgerAccount', amount: Decimal):
        """
        Executes a thread-safe transfer to a target account, enforcing business invariants.
        Prevents mismatched currencies and double-debiting.
        """
        if not target or amount is None:
            raise ValueError("Target and amount cannot be null")
        
        # PRE-CONDITION ENFORCEMENT: Currency matching
        if self.currency != target.currency:
            raise ValueError(f"Cannot transfer between mismatched currencies: {self.currency} and {target.currency}")

        # PRE-CONDITION ENFORCEMENT: Self-transfer check
        if self.account_id == target.account_id:
            raise ValueError("Cannot transfer to the same account")

        # To prevent deadlocks, lock accounts in a stable global order
        locks = [self, target]
        locks.sort(key=lambda acc: acc.account_id)

        with locks[0]._lock:
            with locks[1]._lock:
                # Execute atomic debit-credit sequence
                self.debit(amount)
                target.credit(amount)
```
