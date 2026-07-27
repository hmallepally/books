```python
from decimal import Decimal
from uuid import UUID
import threading

class LedgerAccount:
    """
    Represents a stateful Ledger Account in AuraPay, enforcing business invariants
    during state transitions.
    """
    def __init__(self, account_id: UUID, currency: str, initial_balance: Decimal, overdraft_limit: Decimal):
        if not account_id or not currency:
            raise ValueError("Account ID and Currency cannot be null")
        if initial_balance is None or overdraft_limit is None:
            raise ValueError("Initial balance and overdraft limit cannot be null")
        if overdraft_limit < 0:
            raise ValueError("Overdraft limit cannot be negative")
        if initial_balance + overdraft_limit < 0:
            raise ValueError("Initial balance violates the overdraft limit")

        self.account_id = account_id
        self.currency = currency
        self._balance = initial_balance
        self.overdraft_limit = overdraft_limit
        self._lock = threading.Lock()

    @property
    def balance(self) -> Decimal:
        with self._lock:
            return self._balance

    def credit(self, amount: Decimal):
        """Credits the account. Enforces positive credit amount."""
        if amount is None or amount <= 0:
            raise ValueError("Credit amount must be positive")
        with self._lock:
            self._balance += amount

    def debit(self, amount: Decimal):
        """Debits the account. Enforces balance invariants and overdraft limits."""
        if amount is None or amount <= 0:
            raise ValueError("Debit amount must be positive")
        
        with self._lock:
            new_balance = self._balance - amount
            # INVARIANT ENFORCEMENT
            if new_balance + self.overdraft_limit < 0:
                raise ValueError(
                    f"Debit of {amount} exceeds account overdraft boundary. "
                    f"Balance: {self._balance}, Limit: -{self.overdraft_limit}"
                )
            self._balance = new_balance
```
