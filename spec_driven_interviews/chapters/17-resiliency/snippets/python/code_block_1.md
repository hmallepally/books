```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

@dataclass
class OutboxEvent:
    id: UUID
    aggregate_type: str
    aggregate_id: UUID
    event_type: str
    payload: str
    created_at: datetime
    processed: bool

class MessageBrokerClient(ABC):
    @abstractmethod
    def publish(self, topic: str, payload: str):
        pass

class OutboxRepository(ABC):
    @abstractmethod
    def find_unprocessed_and_lock(self, limit: int) -> List[OutboxEvent]:
        pass

    @abstractmethod
    def mark_as_processed(self, event_id: UUID):
        pass

class TransactionalOutboxPublisher:
    """
    Service that polls the database Outbox table and publishes events to the broker.
    Guarantees At-Least-Once delivery of domain events.
    """
    def __init__(self, outbox_repository: OutboxRepository, broker_client: MessageBrokerClient):
        self.outbox_repository = outbox_repository
        self.broker_client = broker_client

    def publish_pending_events(self):
        # Retrieve unprocessed events under lock
        pending_events = self.outbox_repository.find_unprocessed_and_lock(100)

        for event in pending_events:
            try:
                # Publish to broker (external network call)
                topic = f"events.{event.aggregate_type.lower()}"
                self.broker_client.publish(topic, event.payload)

                # Mark as processed in the database
                self.outbox_repository.mark_as_processed(event.id)
            except Exception as e:
                # If publishing fails, we log and skip.
                # It will be retried on the next poll cycle (At-Least-Once).
                print(f"Failed to publish outbox event {event.id}: {str(e)}. Will retry.")
```
