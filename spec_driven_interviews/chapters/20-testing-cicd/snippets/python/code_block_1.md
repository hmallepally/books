```python
import unittest
from unittest.mock import Mock, call

class TestTransactionProcessor(unittest.TestCase):
    def test_successful_transfer_enforces_invariants(self):
        # Arrange Mock Dependencies
        mock_repo = Mock()
        mock_calculator = Mock()
        mock_sender = Mock()

        source = LedgerAccount("acc-source", 100.00, "USD")
        destination = LedgerAccount("acc-dest", 50.00, "USD")

        mock_repo.find_by_id.side_effect = lambda uid: source if uid == "acc-source" else destination
        mock_calculator.calculate_fee.return_value = 0.0

        processor = TransactionProcessor(mock_repo, mock_calculator, mock_sender)

        # Act
        processor.process_transfer("acc-source", "acc-dest", 30.0)

        # Assert state invariants updated
        self.assertEqual(70.0, source.balance)
        self.assertEqual(80.0, destination.balance)

        # Assert repository saved both
        mock_repo.save.assert_has_calls([call(source), call(destination)])
        mock_sender.send_notification.assert_called_once()
```