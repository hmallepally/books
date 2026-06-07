```python
# tests/test_invariants_transfer.py
import pytest
from decimal import Decimal
@pytest.mark.asyncio
async def test_invariant_conservation_of_mass(db_session, user_a, user_b):
    """
    INVARIANT: Sender + Receiver balances must equal the exact same total
    before and after the transaction, factoring in the network fee.
    """
    initial_a = await db_session.get_balance(user_a.id)
    initial_b = await db_session.get_balance(user_b.id)
    initial_total = initial_a + initial_b
    transfer_amount = Decimal("100.00")
    network_fee = Decimal("1.50")
    # Execute Transfer
    await execute_fund_transfer(
        db_session, 
        sender_id=user_a.id, 
        receiver_id=user_b.id, 
        amount=transfer_amount
    )
    final_a = await db_session.get_balance(user_a.id)
    final_b = await db_session.get_balance(user_b.id)
    # The network fee leaves the system, so we add it back to verify conservation of mass
    final_total = final_a + final_b + network_fee
    # THE ABSOLUTE TRUTH
    assert initial_total == final_total, "INVARIANT BREACH: Mass not conserved."
    # Ensure money actually moved correctly
    assert final_a == initial_a - transfer_amount - network_fee
    assert final_b == initial_b + transfer_amount
```