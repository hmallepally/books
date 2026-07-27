```python
from decimal import Decimal
from typing import List, Dict
from uuid import UUID
from collections import defaultdict
from functools import reduce

class TransactionAnalytics:
    """
    Demonstrates high-performance batch transaction analytics in Python.
    """
    def aggregate_merchant_volumes(
        self, 
        transactions: List, 
        min_amount_threshold: Decimal
    ) -> Dict[UUID, Decimal]:
        if transactions is None or min_amount_threshold is None:
            raise ValueError("Transactions and threshold cannot be null")

        # 1. Filter: Retain transactions meeting the value criteria
        filtered_txs = filter(lambda t: t.amount >= min_amount_threshold, transactions)

        # 2. Collect/Reduce: Group by merchant and sum the transaction volume
        merchant_volumes = defaultdict(Decimal)
        for tx in filtered_txs:
            merchant_volumes[tx.destination_account_id] += tx.amount

        return dict(merchant_volumes)

    def get_high_value_transaction_ids(self, transactions: List, limit: Decimal) -> List[UUID]:
        # Declarative list comprehension matching functional map/filter
        return [
            t.transaction_id 
            for t in transactions 
            if t.amount > limit
        ]
```
