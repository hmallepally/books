```python
from confluent_kafka import Producer

class TransactionEventProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        config = {
            'bootstrap.servers': bootstrap_servers,
            'enable.idempotence': True,
            'acks': 'all'
        }
        self.producer = Producer(config)
        self.topic = topic

    def publish_event(self, account_id: str, event_json: str):
        # Shard by account_id to guarantee partition ordering
        self.producer.produce(
            self.topic, 
            key=account_id.encode('utf-8'), 
            value=event_json.encode('utf-8'),
            callback=lambda err, msg: print(f"Published: {msg.key()}") if not err else print(f"Error: {err}")
        )
        self.producer.poll(0)
```