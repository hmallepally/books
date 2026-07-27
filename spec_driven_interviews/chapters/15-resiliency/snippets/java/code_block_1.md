```java
package com.aurapay.integration;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

/**
 * Represents an Outbox event record stored in the same database as the business entities.
 */
public record OutboxEvent(
    UUID id,
    String aggregateType,
    UUID aggregateId,
    String eventType,
    String payload,
    Instant createdAt,
    boolean processed
) {}

/**
 * Interface representing the Message Broker client (e.g., Kafka, RabbitMQ).
 */
interface MessageBrokerClient {
    void publish(String topic, String payload) throws Exception;
}

/**
 * Service that polls the database Outbox table and publishes events to the broker.
 * Guarantees At-Least-Once delivery of domain events.
 */
public class TransactionalOutboxPublisher {
    private final OutboxRepository outboxRepository;
    private final MessageBrokerClient brokerClient;

    public TransactionalOutboxPublisher(OutboxRepository outboxRepository, MessageBrokerClient brokerClient) {
        this.outboxRepository = outboxRepository;
        this.brokerClient = brokerClient;
    }

    /**
     * Polling worker method. In production, this would be executed by a background 
     * scheduler or transaction log tailer (Debezium/CDC).
     */
    public void publishPendingEvents() {
        // Retrieve unprocessed events (locking them to prevent double-processing by other nodes)
        List<OutboxEvent> pendingEvents = outboxRepository.findUnprocessedAndLock(100);

        for (OutboxEvent event : pendingEvents) {
            try {
                // Publish to broker (external network call)
                String topic = "events." + event.aggregateType().toLowerCase();
                brokerClient.publish(topic, event.payload());

                // Mark as processed in the database
                outboxRepository.markAsProcessed(event.id());
            } catch (Exception e) {
                // If publishing fails, we do NOT mark it as processed.
                // It will be retried on the next poll cycle (At-Least-Once Delivery).
                System.err.printf("Failed to publish outbox event %s: %s. Will retry.%n", 
                    event.id(), e.getMessage());
            }
        }
    }
}

/**
 * Interface representing database operations for the Outbox table.
 */
interface OutboxRepository {
    List<OutboxEvent> findUnprocessedAndLock(int limit);
    void markAsProcessed(UUID eventId);
}
```
