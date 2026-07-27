# Distributed Event Streaming and Message Brokers

> *"An event log is the ultimate source of historical truth. In a distributed architecture, message brokers serve as the central nervous system, routing states across service boundaries."*


## Event Streaming in System Design

In microservice architectures, services must communicate asynchronously. Interview candidates often default to saying: *"We will send a message via Kafka."* 

If you stop there, you miss the opportunity to demonstrate depth. A senior systems architect must explain how the message broker is structured, how partition keys guarantee message ordering under concurrency, and how to achieve **Exactly-Once Semantics (EOS)** across transactions.

In this chapter, we deep-dive into Apache Kafka's storage internals and partition routing mechanics, showing how AuraPay shards event streams to maintain ledger correctness.

![Apache Kafka Topic Partitions and Consumer Groups](visuals/kafka_internals.png){width=90%}


## Apache Kafka Internals & Sharding

Apache Kafka is designed as a distributed, partitioned, commit log. Understanding its storage structure is critical for scaling system throughput:

> **Why is it called "Kafka"?** LinkedIn engineer Jay Kreps named it after **Franz Kafka**, the Czech novelist famous for writing about surreal, labyrinthine bureaucracies. Kreps chose the name because Kafka is *"a system optimized for writing"* — and Franz Kafka was a writer. The literary nod is fitting: just as Kafka's novels depict characters navigating complex, opaque systems, Apache Kafka routes millions of messages through complex distributed topologies. The name stuck, and today "Kafka" is synonymous with high-throughput event streaming.

### Core Concepts

1. **The Commit Log:** A Kafka partition is an append-only, ordered sequence of records. Each record consists of a key, a value, and a timestamp. Records are immutable and assigned a sequential ID called an **offset**.
2. **Partitions:** Topics are divided into multiple partitions distributed across Kafka brokers. Partitions are the unit of scalability in Kafka: while a single partition can only handle a throughput limited by its host broker, multiple partitions allow parallel writes and reads across the cluster.
3. **Consumer Groups:** A consumer group is a collection of consumers working together to read messages from a topic. Kafka guarantees that each partition is assigned to exactly *one* consumer instance within a consumer group. This prevents duplicate processing of messages.

### Replication and Durability
Each partition is replicated across multiple brokers for fault tolerance:

- **Leader Replica:** Handles all read and write requests for the partition.
- **Follower Replicas:** Passively replicate data from the leader. If the leader broker fails, a follower is promoted to leader via the controller election process.
- **ISR (In-Sync Replicas):** The set of replicas that are fully caught up with the leader. The producer configuration `acks=all` ensures a write is only acknowledged after all ISR replicas have persisted it, preventing data loss during broker failures.
- **Minimum ISR:** Setting `min.insync.replicas=2` with `acks=all` ensures at least two replicas must acknowledge a write. If only one replica is available, the broker rejects the write rather than risking data loss.


## The Ordering Invariant & Partition Keys

A common failure mode in message broker design is **out-of-order message delivery**. 
For example, if a user performs two actions:

1. Deposit \$100 (Event A)
2. Withdraw \$80 (Event B)

If the consumer processes Event B before Event A due to network concurrency, the withdrawal may be rejected due to insufficient funds, violating the system's business invariants.

### Guaranteeing In-Order Delivery
To guarantee in-order delivery, Kafka enforces a strict rule: **messages written to the same partition are always read in the exact order they were written.**

- If you publish messages without a key (null key), Kafka distributes them across partitions using a round-robin algorithm, losing all ordering guarantees.
- **The Solution:** Publish messages with a **Partition Key** (e.g., `accountId`). Kafka hashes the key to determine the partition:

$$\text{Partition ID} = \text{hash}(\text{accountId}) \pmod{\text{Number of Partitions}}$$

By sharding on `accountId`, all transaction events for a specific account are guaranteed to land in the same partition and be processed in exact chronological order by a single consumer thread.

The following code illustrates this partition-key routing implementation in a Kafka producer:

```csharp
using Confluent.Kafka;
using System.Threading.Tasks;

public class TransactionEventProducer 
{
    private final IProducer<string, string> _producer;
    private final string _topic;

    public TransactionEventProducer(string bootstrapServers, string topic) 
    {
        var config = new ProducerConfig
        {
            BootstrapServers = bootstrapServers,
            EnableIdempotence = true,
            Acks = Acks.All
        };
        _producer = new ProducerBuilder<string, string>(config).Build();
        _topic = topic;
    }

    public async Task PublishEventAsync(string accountId, string eventJson) 
    {
        // Shard by accountId to guarantee partition message ordering
        var message = new Message<string, string> { Key = accountId, Value = eventJson };
        await _producer.ProduceAsync(_topic, message);
    }
}
```

Setting `enable.idempotence = true` ensures that network retries by the producer do not result in duplicate messages landing in the partition log.


## Exactly-Once Semantics (EOS)

In system design interviews, clearing the **Exactly-Once** challenge is a major differentiator.
How do you guarantee that a transaction is processed exactly once, even if the network fails midway?

Kafka achieves Exactly-Once Semantics (EOS) using a combination of three mechanisms:

1. **Idempotent Producers:** The producer appends a unique sequence number to each message. If a broker receives a duplicate sequence number due to a network retry, it discards the duplicate write.
2. **Transactional Writes:** When a service must read a message from an input topic, process it, and write the output to another topic (the Read-Process-Write pattern), Kafka allows wrapping these steps in a transaction.
3. **Offset Commits in DB:** Alternatively, when writing to a database (like AuraPay's ledger), combine the database write and the event offset commit inside a single database transaction. This is the **Transactional Outbox Pattern** (discussed in the Enterprise Integration and Resiliency chapter), ensuring that either both the database update and the event publish succeed, or both roll back.


## Consumer Group Rebalancing and Failure Recovery

When a consumer instance crashes or a new instance joins the group, Kafka triggers a **rebalance** — redistributing partition assignments across the remaining consumers:

### The Rebalancing Problem
During a rebalance, all consumers in the group temporarily stop processing. This "stop-the-world" pause can cause latency spikes in real-time systems.

### Mitigation Strategies

1. **Sticky Assignor:** Use the `StickyAssignor` partition assignment strategy. Unlike the default `RangeAssignor`, it minimizes partition movement during rebalances — consumers keep their existing assignments, and only the partitions owned by the departing consumer are redistributed.
2. **Cooperative Rebalancing:** Kafka 2.4+ supports **incremental cooperative rebalancing**, where only the affected partitions are revoked and reassigned. Non-affected consumers continue processing without interruption.
3. **Static Group Membership:** Assign a fixed `group.instance.id` to each consumer. When a consumer restarts within the `session.timeout.ms` window, Kafka recognizes it as the same member and skips the rebalance entirely.

### Dead Letter Queues (DLQ)
When a consumer repeatedly fails to process a message (e.g., due to a malformed payload or a downstream service outage), it must not block the entire partition:

- After a configurable number of retry attempts (e.g., 3), route the failed message to a **Dead Letter Queue** — a separate Kafka topic (e.g., `transactions.dlq`).
- The main consumer continues processing subsequent messages.
- A separate monitoring service reads the DLQ, alerts the operations team, and supports manual inspection and replay.


## Event Schema Evolution

As your system evolves, the structure of event payloads will change. Adding new fields, renaming properties, or changing data types can break downstream consumers if not managed carefully:

### Schema Registry (Confluent)

- **Central Registry:** All event schemas are registered in a **Schema Registry** (e.g., Confluent Schema Registry) using Avro, Protobuf, or JSON Schema formats.
- **Compatibility Modes:**
  - **BACKWARD:** New schema can read data written with the old schema. Achieved by only adding optional fields with defaults.
  - **FORWARD:** Old schema can read data written with the new schema. Achieved by only removing optional fields.
  - **FULL:** Both backward and forward compatible — the safest option for production systems.
- **Enforcement:** Producers must validate their serialized payload against the registered schema before publishing. If the payload violates the compatibility rules, the write is rejected at the producer level, preventing corrupt data from entering the topic.


## Kafka vs. Event-Driven Alternatives

### When NOT to Use Kafka
Kafka excels at high-throughput, ordered event streaming. However, it is not always the right choice:

- **Simple Task Queues:** If you need to distribute work items across workers without ordering guarantees (e.g., image resizing, email sending), a simpler queue like **RabbitMQ** or **AWS SQS** reduces operational complexity.
- **Real-Time WebSocket Push:** Kafka is pull-based. For real-time push notifications to browsers or mobile clients, use **Redis Pub/Sub** or a dedicated WebSocket gateway.
- **Sub-Millisecond Latency:** Kafka's batching and replication introduce millisecond-range latency. For ultra-low-latency inter-process communication (e.g., inside a matching engine), use shared memory or in-process queues.

## Message Broker Comparison Matrix

Selecting the right broker technology depends on the architectural requirements:

| Dimension | Apache Kafka | RabbitMQ | AWS SQS / SNS |
|---|---|---|---|
| **Architecture** | Partitioned commit log (pull-based) | Smart broker, dumb consumer (push-based) | Cloud-managed queue (pull/push-based) |
| **Throughput Scale** | **Extreme** (10M+ messages/sec via sequential disc I/O) | High (Capped by broker memory and queue routing complexity) | High (Managed automatically by cloud scaling limits) |
| **Routing Capability** | Basic (consumer groups read whole topic partitions) | **Complex** (supports exchange routing keys, fanout, headers) | Basic (SNS topic fanout to SQS queues) |
| **Ordering Guarantees** | Strict order *per partition* via keys | Order guaranteed only for single consumers | Strict order only when utilizing FIFO queues (low throughput) |
| **Backpressure** | Managed by consumer (consumer pulls when ready) | Smart broker manages queue size (pushes back on producer) | Managed by consumer pooling configurations |
| **Retention** | Durable (retains messages on disk for days/weeks) | Transient (messages are deleted immediately after consumption) | Transient (messages deleted after poll commit, max 14 days) |
| **Schema Evolution** | Schema Registry (Avro/Protobuf) | No native schema support | No native schema support |


> ⭐ **STAR Moment: The Ordering Guarantee**
> 
> In a system design interview, explain: *"We will configure our payment topics with a partitioning key based on the ledger account ID. This guarantees that all transactions affecting a specific account are processed sequentially by a single thread in our consumer group, eliminating race conditions and balance corruption during high-frequency parallel events. We use the StickyAssignor with cooperative rebalancing to minimize processing pauses when consumers scale, and route poison messages to a Dead Letter Queue after three retry attempts to prevent partition blocking."* This shows deep understanding of partition routing, failure recovery, and operational maturity.
