```python
import unittest
from unittest.mock import Mock, ANY
from decimal import Decimal

class TestTransactionProcessor(unittest.TestCase):
    def test_successful_transfer_enforces_invariants(self):
        # Arrange Mock Dependencies
        mock_repo = Mock()
        mock_calculator = Mock()
        mock_sender = Mock()

        source = LedgerAccount("acc-source", Decimal("100.00"), "USD")
        destination = LedgerAccount("acc-dest", Decimal("50.00"), "USD")

        mock_repo.find_by_id.side_effect = lambda id: source if id == "acc-source" else destination
        mock_calculator.calculate_fee.return_value = Decimal("0.00")

        processor = TransactionProcessor(mock_repo, mock_calculator, mock_sender)

        # Act
        processor.process_transfer("acc-source", "acc-dest", Decimal("30.00"))

        # Assert state invariants updated
        self.assertEqual(Decimal("70.00"), source.get_balance())
        self.assertEqual(Decimal("80.00"), destination.get_balance())

        # Assert repository saved both
        mock_repo.save.assert_any_call(source)
        mock_repo.save.assert_any_call(destination)
        mock_sender.send_notification.assert_called_with(ANY)
```