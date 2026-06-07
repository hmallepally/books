```python
from hypothesis import given, strategies as st
from decimal import Decimal
import pytest

# Invariant: Sender Balance + Receiver Balance must equal initial total (minus network fee)
@given(
    transfer_amount=st.decimals(min_value=0.01, max_value=1000000.00, places=2),
    initial_a=st.decimals(min_value=0.00, max_value=2000000.00, places=2),
    initial_b=st.decimals(min_value=0.00, max_value=2000000.00, places=2)
)
def test_conservation_of_mass(transfer_amount, initial_a, initial_b):
    # Setup state
    account_a = Account(balance=initial_a)
    account_b = Account(balance=initial_b)
    network_fee = Decimal("1.50")

    # Execute transfer
    try:
        process_transfer(account_a, account_b, transfer_amount, network_fee)

        # Verify Invariant explicitly
        final_total = account_a.balance + account_b.balance + network_fee
        assert final_total == (initial_a + initial_b)
    except InsufficientFundsError:
        # If the transfer fails, balances must remain completely untouched
        assert account_a.balance == initial_a
        assert account_b.balance == initial_b
```