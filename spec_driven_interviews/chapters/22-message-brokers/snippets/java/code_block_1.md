```java
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import java.util.Properties;

public class TransactionEventProducer {
    private final KafkaProducer<String, String> producer;
    private final String topic;

    public TransactionEventProducer(String bootstrapServers, String topic) {
        Properties props = new Properties();
        props.put("bootstrap.servers", bootstrapServers);
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        // Guarantee exactly-once idempotency
        props.put("enable.idempotence", "true");
        props.put("acks", "all");

        this.producer = new KafkaProducer<>(props);
        this.topic = topic;
    }

    public void publishEvent(String accountId, String eventJson) {
        // Shard by accountId (key) to guarantee in-order processing per partition
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, accountId, eventJson);
        producer.send(record, (metadata, exception) -> {
            if (exception != null) {
                log.error("Failed to publish event for account: " + accountId, exception);
            }
        });
    }
}
```