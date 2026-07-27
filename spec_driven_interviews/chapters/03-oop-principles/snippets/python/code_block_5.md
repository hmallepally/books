```python
# Anemic Account Model (Fragile Data Holder)
class Account:
    def __init__(self, id: str, balance: float, currency: str):
        self.id = id
        self.balance = balance
        self.currency = currency

# Stateless Service containing business invariants (Anti-pattern)
class LedgerService:
    def transfer(self, from_acc: Account, to_acc: Account, amount: float) -> None:
        if from_acc.balance < amount:
            raise ValueError("Insufficient funds")
        if from_acc.currency != to_acc.currency:
            raise ValueError("Currency mismatch")
        from_acc.balance -= amount
        to_acc.balance += amount
```