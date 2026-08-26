```csharp
using System;
using System.Collections.Generic;

namespace AuraPay.Integration
{
    public record OutboxEvent(
        Guid Id,
        string AggregateType,
        Guid AggregateId,
        string EventType,
        string Payload,
        DateTime CreatedAt,
        bool Processed
    );

    public interface IMessageBrokerClient
    {
        void Publish(string topic, string payload);
    }

    public interface IOutboxRepository
    {
        List<OutboxEvent> FindUnprocessedAndLock(int limit);
        void MarkAsProcessed(Guid eventId);
    }

    /// <summary>
    /// Service that polls the database Outbox table and publishes events to the broker.
    /// Guarantees At-Least-Once delivery of domain events.
    /// </summary>
    public class TransactionalOutboxPublisher
    {
        private readonly IOutboxRepository _outboxRepository;
        private readonly IMessageBrokerClient _brokerClient;

        public TransactionalOutboxPublisher(IOutboxRepository outboxRepository, IMessageBrokerClient brokerClient)
        {
            _outboxRepository = outboxRepository;
            _brokerClient = brokerClient;
        }

        public void PublishPendingEvents()
        {
            // Retrieve unprocessed events under lock
            var pendingEvents = _outboxRepository.FindUnprocessedAndLock(100);

            foreach (var @event in pendingEvents)
            {
                try
                {
                    // Publish to broker (external network call)
                    string topic = $"events.{@event.AggregateType.ToLower()}";
                    _brokerClient.Publish(topic, @event.Payload);

                    // Mark as processed in the database
                    _outboxRepository.MarkAsProcessed(@event.Id);
                }
                catch (Exception e)
                {
                    // If publishing fails, we do NOT mark it as processed.
                    // It will be retried on the next poll cycle (At-Least-Once).
                    Console.Error.WriteLine($"Failed to publish outbox event {@event.Id}: {e.Message}. Will retry.");
                }
            }
        }
    }
}
```
