```python
from decimal import Decimal
from uuid import UUID

class AccountEntity:
    """
    Represents a database-mapped Ledger Account Entity with versioning for
    Optimistic Concurrency Control (OCC).
    """
    def __init__(self, account_id: UUID, balance: Decimal, currency: str, version: int):
        self.id = account_id
        self.balance = balance
        self.currency = currency
        self.version = version

    def update_balance(self, new_balance: Decimal):
        self.balance = new_balance

    def increment_version(self):
        self.version += 1

class DatabaseLedgerRepository:
    """
    Repository implementation executing the version check update query.
    """
    def save(self, account: AccountEntity):
        # Simulates SQL execution:
        # UPDATE accounts SET balance = ?, version = version + 1 WHERE id = ? AND version = ?;
        sql_query = (
            "UPDATE accounts SET balance = :balance, version = :version + 1 "
            "WHERE id = :id AND version = :version"
        )
        
        rows_updated = self._mock_execute_query(sql_query, account)

        # OCC FAILURE CHECK: No rows updated implies a version conflict
        if rows_updated == 0:
            raise RuntimeError(
                f"Optimistic lock conflict on account {account.id}. "
                f"Outdated version: {account.version}"
            )
            
        account.increment_version()

    def _mock_execute_query(self, query: str, account: AccountEntity) -> int:
        # Simulates the database driver execution
        return 1  # 1 indicates success; 0 indicates a version mismatch conflict
```
