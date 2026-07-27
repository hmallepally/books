```python
from enum import Enum

class Side(Enum):
    BUY = 0
    SELL = 1

class Order:
    def __init__(self, id: str, instrument_id: str, side: Side, price: int, quantity: int):
        if price <= 0:
            raise ValueError("Price must be positive")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.id = id
        self.instrument_id = instrument_id
        self.side = side
        self.price = price # Fixed-point integer
        self.quantity = quantity
```