# Master System Design Solutions & Architectural Blueprints

> *"Senior system design is not about guessing technology names; it is the discipline of decomposing complex domain requirements into resilient, mathematically bounded distributed architectures."*

In the preceding chapter, we established the foundational principles of system design: Domain-Driven Design (DDD) bounded contexts, monolithic vs. microservices trade-offs, consistent hash sharding, CQRS, CAP theorem trade-offs, rate limiting, and caching architectures.

This chapter provides **14 Master End-to-End System Design Solutions**. Each solution represents a complete, production-grade architectural blueprint designed to answer real-world senior and staff engineering interview prompts across financial infrastructure, security, social platforms, geospatial dispatch, AI/ML, cloud storage, search engines, real-time messaging, task scheduling, collaborative editors, time-series observability, notification systems, and booking inventory management.


## The 7-Part Architecture Blueprint

To ensure complete clarity and zero ambiguity, every system design solution in this chapter follows a standardized **7-Part Architecture Blueprint**:

1. **Problem Statement & SLAs:** Precise functional requirements and quantitative non-functional SLAs (QPS, latency $p99$, availability, consistency).
2. **Capacity Estimation & Hardware Math:** First-principles mathematical derivations for network ingress bandwidth, memory footprints, and daily/annual disk storage.
3. **Visual Architecture Blueprint:** High-resolution structural diagrams illustrating Gateways, Load Balancers, Worker Pools, In-Memory Caches, Message Brokers, and Databases.
4. **Architectural Workflow & Mechanics:** Deep technical walkthrough of subsystem interactions, fault isolation boundaries, concurrency controls, and state pipelines.
5. **API Contracts & Interface Specs:** Production-grade REST JSON DTOs or gRPC Protobuf definitions.
6. **Database Schema & Data Model:** Relational PostgreSQL DDL, NoSQL Document Schema, or Spatial H3 Index structures.
7. **Execution Sequence & Staff Verbalization:** Step-by-step write/read paths, failure compensation, and high-scoring 45-minute verbalization scripts.


## Master System Design Solutions Catalog

| Solution | System Design Case Study | Primary Architectural Patterns | Generated Diagram Asset |
| :--- | :--- | :--- | :--- |
| **Solution 1** | **AuraPay:** Distributed Global Payment Gateway & Ledger | Idempotency Keys, Double-Entry SQL DDL, Transactional Outbox, Saga Orchestration | `visuals/arch_payment_gateway.png` |
| **Solution 2** | **ZenithTrade:** High-Frequency Order Matching Exchange | In-Memory OrderBook, Raft Consensus, Write-Ahead Log (WAL), CQRS Read Projections | `visuals/arch_matching_engine.png` |
| **Solution 3** | **ChiramTrust:** Distributed Rate Limiter & Fraud Pipeline | Redis Sliding Window Lua script, eBPF Kernel probes, Real-Time ML scoring, SOAR dynamic blocking | `visuals/arch_rate_limiter_fraud.png` |
| **Solution 4** | **Consumer Social:** Real-Time Social Feed & Video Streaming | Hybrid Push/Pull Timeline (Celebrity vs Regular), Redis Sorted Sets, S3 HLS/DASH Transcoding | `visuals/arch_social_video_platform.png` |
| **Solution 5** | **Geospatial:** Real-Time Ride-Sharing Dispatch (Uber/Lyft) | Uber H3 Hexagonal Spatial Indexing, QuadTrees, 3.3M QPS WebSocket ingest, Dynamic Surge Engine | `visuals/arch_rideshare_geospatial.png` |
| **Solution 6** | **AI Infrastructure:** Distributed Vector Search & RAG Engine | HNSW Vector Indexing (Milvus), Sparse BM25 (Elasticsearch), Reciprocal Rank Fusion, LLM Context Assembly | `visuals/arch_vector_rag_system.png` |
| **Solution 7** | **Cloud Storage:** Distributed File Sync Engine (Google Drive) | Rabin Fingerprint Chunking (4MB), Content-Addressable Block Store (S3), Vector Clock Sync | `visuals/arch_drive_sync_storage.png` |
| **Solution 8** | **Search Engine:** Distributed Web Crawler & Search Indexer | URL Frontier (Politeness Queue), SimHash Deduplication, Inverted Index Posting Lists, PageRank Graph | `visuals/arch_web_crawler_search.png` |
| **Solution 9** | **Real-Time Chat:** Distributed Messaging & Presence (Slack/Discord) | WebSocket Gateway, Redis Bitmaps Presence, Cassandra Sequence ID Store, Double Ratchet E2EE | `visuals/arch_chat_messaging_presence.png` |
| **Solution 10** | **Task Scheduler:** Distributed Workflow & Job Engine (Temporal) | Hierarchical Timing Wheel, Task Dependency DAG, Distributed Locks (etcd), Dead Letter Queue | `visuals/arch_task_scheduler_workflow.png` |
| **Solution 11** | **Collaborative Editor:** Real-Time CRDT & Whiteboard (Figma/Docs) | CRDT State Vector Sync, Operational Transformation (OT), Cursor Pub/Sub Stream, Snapshot Engine | `visuals/arch_collaborative_crdt_editor.png` |
| **Solution 12** | **Observability:** Distributed Time-Series Metrics TSDB (Prometheus) | Gorilla Delta-of-Delta Compression, Ring Buffer Chunk Store, Downsampling Aggregator, Alerting Rules | `visuals/arch_metrics_timeseries_observability.png` |
| **Solution 13** | **Notifications:** Multi-Channel Notification & Alerting Platform | Priority Queue Routing, Bloom Filter Deduplication, Channel Adapters (Email/SMS/Push/In-App), Rate Limiting | `visuals/arch_notification_platform.png` |
| **Solution 14** | **Booking Engine:** Distributed Hotel & Flight Inventory System | Redis Redlock Distributed Locks, Reservation Saga, Overbooking Prevention, Calendar Row-Level Locks | `visuals/arch_booking_inventory.png` |


## Master System Design Solutions

### Solution 1: AuraPay — Global Distributed Payment Gateway & Ledger

#### Problem Statement & SLAs
Design a global payment gateway and double-entry ledger capable of processing credit card and bank transactions across international merchants.

- **Target QPS:** 50,000 requests/sec peak.
- **Latency SLA:** $p99 < 150\text{ms}$ end-to-end API response.
- **Consistency SLA:** Strict financial consistency ($0$ double-spending, $0$ lost ledger entries).

#### Capacity Estimation & Hardware Math
- **Traffic Ingress:** $50,000 \text{ QPS} \times 1 \text{ KB payload} = 50 \text{ MB/sec} = 400 \text{ Mbps}$ network ingress.
- **Transaction Volume:** $50,000 \text{ tx/sec} \times 86,400 \text{ sec/day} = 4.32 \text{ billion tx/day}$.
  - Daily storage: $4.32 \times 10^9 \times 500 \text{ bytes} \approx 2.16 \text{ TB/day}$.
  - Annual storage: $\approx 788 \text{ TB/year}$.

#### Concrete Production Hardware & Cluster Topology

| Subsystem Component | AWS Instance Family & Count | Compute & Memory Spec | Primary Architectural Responsibility |
| :--- | :--- | :--- | :--- |
| **API Edge Gateways** | 10 $\times$ `c6i.2xlarge` (Envoy) | 8 vCPU, 16 GB RAM per node | TLS termination, WAF inspection, Redis Lua idempotency check |
| **Idempotency Cache** | 6 $\times$ `r6i.xlarge` (3M + 3R) | 32 GB RAM per node | Distributed Redis Cluster with AOF persistence & sub-millisecond keys |
| **Payment Service Pods**| 40 $\times$ `c6i.4xlarge` (EKS) | 16 vCPU, 32 GB RAM per pod | Stateless transaction coordination, JSON/ISO-8583 bank translation |
| **Relational Database** | 1 Primary + 2 Multi-AZ Replicas | `db.r6i.8xlarge` (256 GB RAM) | 20,000 Provisioned IOPS (io2), Transactional Outbox, Monthly Partitioning |
| **Kafka Broker Cluster**| 6 $\times$ `i3en.2xlarge` (KRaft) | 8 vCPU, 64 GB RAM, NVMe | 2.5 TB NVMe SSD per node, zero-copy DMA streaming, `min.insync.replicas=2` |

#### Visual Architecture Blueprint
![Figure 17.1: AuraPay Payment Gateway & Ledger Architecture](visuals/arch_payment_gateway.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Edge Ingress & Fast Idempotency (API Gateway):**
   - Intercepts incoming payment requests bearing an `Idempotency-Key` header.
   - Queries a distributed fast store (Redis) to verify request state. If the key exists and is `COMPLETED`, the cached response is served immediately. If `PENDING`, concurrent duplicate calls are rejected with `409 Conflict`.
2. **Synchronous Payment Processing:**
   - Forwards brand-new requests to the **Payment Processing Service**, which initiates an authorization call via the **Bank Adapter Service** (translating REST/JSON to legacy ISO 8583 / FIX protocols).
3. **Transactional Outbox Pattern (Dual-Write Prevention):**
   - The Payment Processing Service writes the updated payment entity and emits an outbox event into a single local relational database (**PostgreSQL**) within an atomic `BEGIN ... COMMIT` boundary.
   - An asynchronous relay worker (or CDC pipeline like Debezium) tails the outbox table via Postgres logical decoding (`pgoutput`) and reliably publishes messages (`PaymentCreated`, `PaymentAuthorized`) to **Apache Kafka**.
4. **Decoupled Asynchronous Settlement & Double-Entry Ledger:**
   - The **Saga Orchestrator** consumes events from Kafka and coordinates the multi-step transaction.
   - Dispatches strict double-entry accounting commands to the **Ledger Service** (recording balanced immutable `DEBIT` and `CREDIT` rows with high-precision `NUMERIC(18, 4)` types).
5. **Compensating Actions:**
   - If downstream ledger validation or settlement fails, the Saga Orchestrator executes compensating transactions, marks the Redis idempotency key as `FAILED`, and issues webhook failure alerts.

#### API Contracts & Interface Specs
```json
// POST /v1/payments (HTTP REST / Idempotency Protected)
Header: Idempotency-Key: "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
{
  "account_id": "acc_usr_99812",
  "merchant_id": "mch_stripe_001",
  "amount": 14999,
  "currency": "USD",
  "payment_method_token": "tok_visa_4412"
}
```

#### Production Database Schema & Data Model (PostgreSQL DDL)
```sql
-- Partitioned master ledger table to sustain 2.16 TB daily write volume
CREATE TABLE ledger_entries (
    entry_id UUID NOT NULL,
    transaction_id UUID NOT NULL,
    account_id UUID NOT NULL,
    entry_type VARCHAR(10) NOT NULL CHECK (entry_type IN ('DEBIT', 'CREDIT')),
    amount NUMERIC(18, 4) NOT NULL CHECK (amount > 0),
    currency VARCHAR(3) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entry_id, created_at)
) PARTITION BY RANGE (created_at);

-- Monthly partition tables (automated via pg_partman)
CREATE TABLE ledger_entries_2026_09 PARTITION OF ledger_entries
    FOR VALUES FROM ('2026-09-01 00:00:00+00') TO ('2026-10-01 00:00:00+00');

-- Composite index optimized for backwards chronological ledger statement queries
CREATE INDEX idx_ledger_account_chronological 
    ON ledger_entries(account_id, created_at DESC);

-- Transaction lookup index for reconciliation audits
CREATE INDEX idx_ledger_transaction 
    ON ledger_entries(transaction_id);
```

#### Step-by-Step Execution Sequence
1. **Ingest & Idempotency Check:** API Gateway intercepts request, checks Redis for `Idempotency-Key`. If present and `COMPLETED`, returns cached response. If new, sets `PENDING` with 120-second TTL.
2. **Payment Processing:** Payment Service authorizes funds via external Bank Adapter.
3. **Transactional Outbox:** Payment Service writes transaction record AND an Outbox event into PostgreSQL in a single local ACID transaction.
4. **Asynchronous Ledger Event:** Outbox Worker relays `PaymentAuthorized` event to Kafka topic (`payments.settlement`).
5. **Saga Orchestration:** Saga Orchestrator consumes event, executes double-entry debit/credit commits in Ledger DB, and updates status to `COMPLETED` in Redis.
6. **Failure Compensation:** If bank authorization fails or ledger constraint is violated, Saga Orchestrator publishes a `PaymentFailed` event, reverses any provisional ledger entries, updates the idempotency key to `FAILED`, and triggers a webhook notification to the merchant.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Concurrent Duplicate Requests** | User double-clicks "Pay"; two requests with identical `Idempotency-Key` hit different gateway nodes within $5\text{ms}$. | **Atomic Redis `SETNX` with Lease:** First request acquires lock key with `PENDING` state and 120s TTL. Second request fails `SETNX`, receiving immediate `409 Conflict (Request in progress)`. |
| **Bank Network Timeout** | Bank charges card, but network connection drops before HTTP 200 reaches Payment Service. | **Idempotent Reconciliation Poller:** Payment Service marks state `INDETERMINATE`. Background worker queries bank status endpoint using the original `merchant_transaction_id` before retrying or refunding. |
| **Outbox Worker Crash** | Worker dies after reading outbox table but before publishing to Kafka. | **At-Least-Once Replay with Debezium CDC:** New worker restarts from last committed PostgreSQL LSN (Log Sequence Number). Downstream consumers enforce deduplication via `transaction_id` unique constraints. |
| **Ledger DB Primary Failover** | PostgreSQL primary hardware fails mid-transaction. | **Multi-AZ Synchronous Replication:** AWS Aurora automatically promotes standby replica within $\le 30\text{ seconds}$. Application connection pool (HikariCP) re-establishes connections and retries in-flight transactions. |

#### Staff-Level Interview Verbalization
> *"In designing AuraPay, we enforce two critical invariants: API idempotency via Redis atomic locks, and financial double-entry balance preservation via the Transactional Outbox pattern. By decoupling bank network authorization from ledger settlement using Kafka, we guarantee that database write latencies never block the client response path. Under failure conditions, such as network timeouts during bank authorization, our background reconciliation workers resolve indeterminate states asynchronously using the merchant transaction reference, guaranteeing exactly-once accounting semantics."*


### Solution 2: ZenithTrade — High-Frequency Order Matching Exchange

#### Problem Statement & SLAs
Design a high-frequency cryptocurrency and equity order matching exchange.

- **Target Throughput:** 100,000 orders/sec peak per partition.
- **Latency SLA:** Sub-millisecond matching latency ($p99 < 1\text{ms}$).
- **Availability:** $99.999\%$ uptime with sub-second active-passive failover.

> **Why Single-AZ Raft?** Cross-AZ Raft round-trips add 1–5ms network latency, violating the sub-millisecond SLA. The matching engine Raft cluster is co-located within a single Availability Zone using kernel bypass (DPDK) and NVMe direct I/O. Cross-region disaster recovery uses asynchronous WAL shipping rather than synchronous Raft.

#### Capacity Estimation & Hardware Sizing
- **Order Payload:** 200 bytes per order.
- **Network Bandwidth:** $100,000 \text{ QPS} \times 200 \text{ B} = 20 \text{ MB/sec} = 160 \text{ Mbps}$.
- **In-Memory OrderBook Memory:** $10,000,000 \text{ active open orders} \times 128 \text{ B/order} \approx 1.28 \text{ GB RAM}$ per instrument. Fits comfortably in RAM.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Order Gateway / Routers** | 12 $\times$ `c6i.4xlarge` (EKS) | 16 vCPU, 32 GB RAM, 12.5 Gbps | Consistent hash routing by `instrument_id`, TLS termination, FIX/gRPC protocol decode |
| **In-Memory Matching Engine** | 4 $\times$ `c6i.8xlarge` (Dedicated) | 32 vCPU pinned, 64 GB RAM, DPDK | Single-writer LMAX ring buffer, NUMA-socket pinned, zero lock contention |
| **Sequencer / Raft Group** | 3 $\times$ `i3en.3xlarge` (1 Leader + 2 Hot Standby) | 12 vCPU, 96 GB RAM, NVMe Direct | 7.5 TB NVMe SSD direct I/O (`O_DIRECT`), sub-100$\mu$s sequential WAL flush |
| **Market Data Streaming** | 8 $\times$ `i3en.2xlarge` (KRaft Kafka) | 8 vCPU, 64 GB RAM, NVMe | Zero-copy DMA streaming of tick-by-tick order book depth and trade executions |
| **Historical & Audit Store** | 6 $\times$ `r6i.2xlarge` (Distributed SQL) | 16 vCPU, 128 GB RAM | CockroachDB/TimescaleDB for trade settlement reconciliation and regulatory audit |

#### Visual Architecture Blueprint
![Figure 17.2: ZenithTrade High-Frequency Order Matching Architecture](visuals/arch_matching_engine.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Deterministic Order Partitioning:**
   - A **Consistent Hash Ring Router** inspects the incoming `instrument_id` (e.g., `BTC-USD`, `ETH-USD`) and routes the order to the designated shard/partition, preventing cross-symbol lock contention.
2. **Single-AZ In-Memory Matching Engine (Raft Group):**
   - To strictly enforce sub-millisecond execution ($p99 < 1\text{ms}$), Raft consensus groups are co-located within a single Availability Zone. This bypasses cross-AZ network round-trips ($1\text{--}5\text{ms}$).
   - Employs **DPDK (Data Plane Development Kit)** for kernel-bypass networking and direct NVMe I/O.
   - The **Leader Node** updates in-memory limit order books (LOB) and commits sequential operations to an append-only Write-Ahead Log (WAL).
   - Hot standby **Follower Nodes** replicate the WAL for immediate active-passive failover.
3. **CQRS & Downstream Projections:**
   - Trade executions bypass disk bottlenecks on the read path via CQRS projections.
   - Matched trades stream through **Apache Kafka** out to **Redis** (for real-time order-book dashboards and ticker feeds) and **Elasticsearch** (for historical trade analytics, regulatory compliance, and user trade history).
4. **Disaster Recovery (DR):**
   - Asynchronous WAL shipping replicates state across geographically distinct regions without blocking the critical matching path.

#### In-Memory Data Structure & Mechanical Sympathy
High-frequency trading engines cannot afford heap allocations, dynamic resizing, or lock contention during matching:

1. **Limit Order Book (Price-Time Priority):**
   - **Price Ladder:** `TreeMap<Long, DoublyLinkedList<OrderNode>>` where the key is a discrete 64-bit integer price tick (eliminating floating-point precision hazards). Bids are sorted in descending order (`reverseOrder()`), while Asks are sorted in ascending order.
   - **Queue Head Execution:** Orders at the best bid/ask execute in $\mathcal{O}(1)$ time against incoming market orders.
2. **Instantaneous $\mathcal{O}(1)$ Order Cancellations:**
   - Instead of scanning the price list ($\mathcal{O}(N)$), the engine maintains a direct pointer map: `HashMap<UUID, OrderNode>`.
   - Each `OrderNode` maintains explicit `.prev` and `.next` pointers within its price bucket. An incoming `CancelOrder` unlinks the node in $\mathcal{O}(1)$ constant time:
```text
     node.prev.next = node.next;
     node.next.prev = node.prev;
     ```

3. **LMAX Disruptor Lock-Free Ring Buffer:**
   - Ingress orders enter a pre-allocated circular ring buffer ($2^{20} = 1,048,576$ slots) indexed via bitwise masking: `sequence & (BUFFER_SIZE - 1)`.
   - Uses memory barriers (`VarHandle` / CAS) rather than OS mutexes. Cache lines are padded with 56 dummy bytes (`@Contended`) around the sequence counter to prevent CPU L1/L2 false sharing.

#### API Contracts & Interface Specs (gRPC Protobuf)
```protobuf
syntax = "proto3";
package zenithtrade;

message PlaceOrderRequest {
    string idempotency_key = 1;
    string account_id = 2;
    string instrument_id = 3; // e.g., "BTC-USD"
    enum Side { BUY = 0; SELL = 1; }
    Side side = 4;
    enum OrderType { LIMIT = 0; MARKET = 1; POST_ONLY = 2; }
    OrderType order_type = 5;
    int64 price_in_cents = 6;      // Discrete price tick
    int64 quantity_in_satoshis = 7; // Discrete base unit quantity
    int64 client_timestamp_ns = 8;
}

message ExecutionReport {
    string execution_id = 1;
    string order_id = 2;
    string instrument_id = 3;
    enum Status { NEW = 0; PARTIALLY_FILLED = 1; FILLED = 2; CANCELLED = 3; REJECTED = 4; }
    Status status = 4;
    int64 executed_price = 5;
    int64 executed_quantity = 6;
    int64 leaves_quantity = 7;
    int64 engine_sequence_num = 8;
    int64 match_timestamp_ns = 9;
}
```

#### Production Database Schema & Sequencer State (PostgreSQL DDL)
```sql
-- Historical order audit store (CockroachDB / TimescaleDB)
CREATE TABLE orders (
    order_id UUID NOT NULL,
    sequence_num BIGINT NOT NULL,
    instrument_id VARCHAR(16) NOT NULL,
    account_id UUID NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price_cents BIGINT NOT NULL,
    original_quantity BIGINT NOT NULL,
    leaves_quantity BIGINT NOT NULL,
    status VARCHAR(16) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_id, sequence_num)
) PARTITION BY RANGE (created_at);

-- Executed trades ledger for clearing and settlement
CREATE TABLE trade_executions (
    execution_id UUID NOT NULL,
    sequence_num BIGINT NOT NULL,
    instrument_id VARCHAR(16) NOT NULL,
    buy_order_id UUID NOT NULL,
    sell_order_id UUID NOT NULL,
    buy_account_id UUID NOT NULL,
    sell_account_id UUID NOT NULL,
    fill_price_cents BIGINT NOT NULL,
    fill_quantity BIGINT NOT NULL,
    match_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_id, execution_id)
);

CREATE INDEX idx_trades_buy_acc ON trade_executions(buy_account_id, match_timestamp DESC);
CREATE INDEX idx_trades_sell_acc ON trade_executions(sell_account_id, match_timestamp DESC);
```

#### Step-by-Step Execution Sequence
1. **Instrument Ingress & Routing:** Gateway validates FIX/gRPC payload, tags order with client arrival timestamp, and routes to partition leader via consistent hashing on `instrument_id`.
2. **Monotonic Sequencer & WAL Log:** Leader assigns strict monotonically increasing `sequence_num` and streams entry to NVMe append-only log via zero-copy direct I/O (`O_DIRECT`).
3. **In-Memory Order Matching:**
   - Single-threaded matching loop extracts order from Disruptor ring buffer.
   - Evaluates opposing side tree head (`bids.firstEntry()` or `asks.firstEntry()`).
   - If price crosses, executes trade, decrements `leaves_quantity`, and updates `OrderNode`.
   - If unmatched remainder exists, inserts node at tail of designated price queue and registers pointer in `HashMap<UUID, OrderNode>`.
4. **Asynchronous CQRS & Market Broadcast:** Trade execution emits to Kafka. Downstream consumers project market depth updates to Redis and update WebSocket ticker feeds.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Raft Leader Hardware Crash** | Leader dies while 10,000 in-flight orders are queued in memory. | **Deterministic Standby Promotion:** Raft follower detects missing heartbeat within $150\text{ms}$. Standby node has mirrored the exact same committed WAL; promotion takes $\le 300\text{ms}$ with zero dropped or reordered state. |
| **Network Partition / Split-Brain** | Network isolations create two sub-clusters, each attempting to match orders for `BTC-USD`. | **Strict Raft Quorum Fencing:** Leader requires synchronous ACK from $\ge 2$ of 3 nodes before committing sequence numbers. Partitioned minority leader cannot commit orders and self-fences immediately. |
| **Slow Consumer on Tick Feed** | High-volume market burst blocks WebSocket output buffer, threatening matching loop latency. | **Ring Buffer Drop Policy & Coalesced Deltas:** Matching loop never blocks on broadcast. Market data gateway drops intermediate L2 order book updates and delivers only coalesced snapshot states to lagging subscribers. |
| **State Desynchronization on Replay** | Standby replaying from snapshot diverges due to non-deterministic clock access. | **Pure Deterministic State Machine:** Order processing relies strictly on sequencer-injected timestamp and sequence numbers. No `System.currentTimeMillis()` or randomized logic inside the matching loop. |

#### Staff-Level Interview Verbalization
> *"ZenithTrade achieves sub-millisecond matching latency by eliminating lock contention and kernel overhead. We route orders deterministically by instrument ID to single-threaded in-memory matching cores pinned to physical NUMA sockets, utilizing an LMAX Disruptor lock-free ring buffer. Order books combine tree-based price ladders with direct hash indexes for $\mathcal{O}(1)$ cancellations. High availability is guaranteed via co-located single-AZ Raft replication with append-only NVMe write-ahead logs, ensuring active-passive failover in under 300 milliseconds without dropping committed trades."*


### Solution 3: ChiramTrust — Distributed Rate Limiter & Real-Time Fraud Pipeline

#### Problem Statement & SLAs
Design an enterprise-grade rate limiter and real-time security fraud detection pipeline.

- **Target Throughput:** 500,000 requests/sec across 50 microservices.
- **Latency SLA:** Rate limiting evaluation $p99 < 2\text{ms}$. Fraud scoring delay $p99 < 50\text{ms}$.
- **Resilience SLA:** Fail-open design; rate-limiter infrastructure failure must never block legitimate traffic.

#### Capacity Estimation & Hardware Sizing
- **Rate Limit Keys:** 100 million active users.
- **Redis Memory:** $100 \times 10^6 \text{ keys} \times 64 \text{ bytes} \approx 6.4 \text{ GB RAM}$. Redis Cluster easily handles state with room for sliding window sorted sets.
- **Telemetry Throughput:** $500,000 \text{ req/sec} \times 256 \text{ B payload} = 128 \text{ MB/sec} \approx 1.02 \text{ Gbps}$.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **API Gateways (Envoy)** | 30 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Edge TLS termination, local eBPF XDP packet filtering, gRPC rate-limit checks |
| **Redis Rate-Limiter Cluster** | 12 $\times$ `r6i.xlarge` (6 Primary + 6 Replica) | 4 vCPU, 32 GB RAM per node | Sharded by `{client_id}` hash tags, sub-millisecond atomic Lua execution, AOF disabled |
| **Kafka Telemetry Stream** | 6 $\times$ `i3en.xlarge` (KRaft) | 4 vCPU, 32 GB RAM, NVMe | Ingests 500k req/sec connection telemetry without backpressuring edge gateways |
| **Fraud Scoring Engine** | 24 $\times$ `c6i.4xlarge` (Ray/Flink Cluster) | 16 vCPU, 32 GB RAM | Real-time feature extraction, ONNX-optimized XGBoost/GBDT inference ($< 15\text{ms}$) |
| **Security Policy & Event DB** | 1 Primary + 2 Replicas | `db.r6i.4xlarge` (128 GB RAM) | Aurora PostgreSQL, monthly partitioned fraud log, dynamic policy distribution |

#### Visual Architecture Blueprint
![Figure 17.3: ChiramTrust Distributed Rate Limiter & Fraud Pipeline](visuals/arch_rate_limiter_fraud.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Low-Latency Edge Rate Limiting:**
   - Built directly into the **API Gateway (NGINX / Envoy)**.
   - Evaluates incoming calls against **Redis Cluster** using atomic Lua scripts. To prevent `CROSSSLOT` errors across sharded clusters, keys use Redis Hash Tags: `{user_id}:rate_limit`.
2. **Volumetric DDoS Mitigation via eBPF XDP:**
   - Volumetric layer-3/4 attacks (e.g., SYN floods, UDP amplification) are mitigated using **eBPF XDP (eXpress Data Path)** hooks loaded directly into the network interface card (NIC) driver.
   - Malicious IP packets are dropped in under $1\mu\text{s}$ before the Linux kernel allocates an `sk_buff` or initiates context switching, preserving gateway CPU.
3. **Asynchronous ML Fraud Inference Pipeline:**
   - High-throughput Kafka topics (`API_GATEWAY_EVENTS`, `NETWORK_TELEMETRY`) feed stream workers that extract dynamic behavioral features (e.g., velocity spikes, geo-hopping, credential stuffing).
   - Real-time models score transactions against fraud heuristics.
4. **Closed-Loop SOAR Feedback:**
   - High-risk fraud scores trigger the **Security Orchestration (SOAR)** platform to dynamically push updated IP blocklists directly into Redis and eBPF kernel maps.
   - Historical logs land in a Data Lake for continuous offline model retraining.

#### Redis Lua Script (Atomic Sliding Window with Hash Tags)
```lua
-- Enforce hash tagging: key must be formatted as {identifier}:rate_limit
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local clearBefore = now - window

-- 1. Remove expired timestamps outside the current window
redis.call('ZREMRANGEBYSCORE', key, 0, clearBefore)

-- 2. Check current window cardinality
local currentRequests = redis.call('ZCARD', key)

if currentRequests < limit then
    -- 3. Add current timestamp with unique member to handle concurrent millisecond requests
    redis.call('ZADD', key, now, now .. '-' .. redis.call('INCR', key .. ':seq'))
    redis.call('EXPIRE', key, math.ceil(window / 1000))
    return 1
else
    return 0
end
```

#### Sliding Window Counter Approximation Formula

To achieve sub-millisecond edge rate limiting without storing individual request timestamps in Sorted Sets, the **Sliding Window Counter** approximates rolling volume using two fixed-window counters:

$$\text{Estimated Count} = M_{\text{current}} + M_{\text{previous}} \times \left(1 - \frac{t - t_{\text{start}}}{W}\right)$$

- $M_{\text{current}}$: Request count in current 60-second window.
- $M_{\text{previous}}$: Request count in previous 60-second window.
- $t - t_{\text{start}}$: Elapsed time within current window (in seconds).
- $W$: Window duration (60 seconds).
- **Accuracy Bound:** Maximum error is strictly bounded below $0.05\%$ under steady traffic, consuming only 16 bytes of RAM per client key (`INCRBY` / `GET`).

> **DDoS Fallback:** Under volumetric attack, `ZREMRANGEBYSCORE` complexity rises to $O(\log N + M)$ where $M$ is evicted elements. If $M$ spikes, fall back to a fixed-window counter (`INCR key; EXPIRE key window`) to protect the single-threaded Redis event loop.

#### API Contracts & Interface Specs
```json
// GET /v1/rate-limit/check
Header: X-Client-IP: "203.0.113.42"
Header: X-Service-ID: "payment-svc"
Response (200 OK):
{
  "allowed": true,
  "remaining": 847,
  "limit": 1000,
  "window_seconds": 60,
  "retry_after_ms": null
}
// Response (429 Too Many Requests):
{
  "allowed": false,
  "remaining": 0,
  "limit": 1000,
  "window_seconds": 60,
  "retry_after_ms": 12400
}
```

#### Production Database Schema & Data Model (PostgreSQL DDL)
```sql
CREATE TABLE rate_limit_policies (
    policy_id UUID PRIMARY KEY,
    service_id VARCHAR(64) NOT NULL,
    endpoint_pattern VARCHAR(255) NOT NULL,
    max_requests INT NOT NULL,
    window_seconds INT NOT NULL,
    burst_multiplier DECIMAL(3,1) DEFAULT 1.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE fraud_events (
    event_id UUID NOT NULL,
    client_ip INET NOT NULL,
    service_id VARCHAR(64) NOT NULL,
    fraud_score DECIMAL(5,4) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    action_taken VARCHAR(20) NOT NULL CHECK (action_taken IN ('ALLOWED', 'THROTTLED', 'BLOCKED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (event_id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE ip_reputation_blacklist (
    cidr_range CIDR PRIMARY KEY,
    reason VARCHAR(128) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_fraud_ip ON fraud_events(client_ip, created_at DESC);
CREATE INDEX idx_policies_service ON rate_limit_policies(service_id, endpoint_pattern);
```

#### Step-by-Step Execution Sequence
1. **Gateway Evaluation:** Envoy API Gateway intercepts request, extracts `{client_id}`, and issues non-blocking gRPC call to Redis Rate Limiter cluster.
2. **Atomic Lua Execution:** Redis executes sliding window script. If count $\le \text{limit}$, returns `allowed=true`. If exceeded, gateway returns `429 Too Many Requests` with `Retry-After`.
3. **eBPF Telemetry Hook:** Linux kernel eBPF probe captures TCP connection metadata without user-space context switching overhead.
4. **Streaming Scoring:** Kernel telemetry streams to Kafka (`telemetry.events`). Real-time ML worker scores fraud probability within 15ms.
5. **SOAR Dynamic Ban:** If fraud score $> 0.90$, automated Security Orchestration (SOAR) updates the Redis block list and injects the CIDR into the eBPF XDP filter map, dropping subsequent packets at wire speed.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Redis Cluster Node Outage** | Master node dies; failover takes 3–5 seconds, during which rate limit checks fail. | **Fail-Open with Circuit Breaker:** Envoy gateway wraps Redis checks in a circuit breaker. If Redis times out ($> 5\text{ms}$), gateway defaults to *Fail-Open* and tracks local in-memory token bucket limits. Legitimate user requests are never dropped. |
| **Distributed Clock Skew** | Clock drift across API gateway nodes causes sliding window timestamps to desynchronize. | **Redis Server-Side Time (`TIME` command):** Timestamp evaluation is derived from the Redis server's monotonic clock (`redis.call('TIME')`) or synchronized via Amazon Time Sync Service (NTP with leap second smoothing), keeping drift $< 1\text{ms}$. |
| **Thundering Herd on Policy Reload** | 30 gateways simultaneously query PostgreSQL on policy cache invalidation, overwhelming DB connections. | **Staggered Jittered Invalidation & Redis Pub/Sub:** Central policy updates publish an invalidation event over Redis Pub/Sub. Gateway instances listen and reload configurations with random exponential jitter ($0\text{--}500\text{ms}$). |
| **Volumetric DDoS Overwhelming Redis** | Massive distributed botnet floods millions of unique IPs, exhausting Redis RAM with sorted sets. | **Tiered Rate Limiting & eBPF XDP Dropping:** Edge Envoy monitors global bandwidth. If QPS exceeds threshold, gateway switches from Sorted Set sliding window to fixed-window counter (`INCR`), and eBPF XDP drops high-rate CIDRs before socket allocation. |

#### Staff-Level Interview Verbalization
> *"Our rate limiting and fraud architecture enforces defense-in-depth across two distinct planes: a synchronous control plane and an asynchronous intelligence plane. On the synchronous path, Envoy gateways execute atomic Redis Lua scripts utilizing hash-tagged keys to achieve sub-2ms rate evaluation, backed by a fail-open circuit breaker. On the asynchronous path, eBPF probes stream kernel connection telemetry to a Kafka and Ray scoring pipeline. When malicious behavior is detected, SOAR dynamically pushes IP blocks down to eBPF XDP driver hooks, dropping attack packets at wire speed without consuming CPU cycles."*


### Solution 4: Consumer Scale — Real-Time Social Feed & Video Streaming Platform

#### Problem Statement & SLAs
Design a consumer social timeline (Twitter/X) and adaptive video streaming platform (YouTube/TikTok).

- **Active Audience:** 300 million daily active users (DAU).
- **Latency SLA:** Timeline generation $p99 < 200\text{ms}$. Video start-to-play $< 1.5\text{s}$ globally.
- **Availability:** $99.99\%$ for feed queries; $99.999\%$ for static video CDN delivery.

#### Capacity Estimation & Hardware Sizing
- **Write QPS (Posts):** $5,000 \text{ posts/sec}$ average, $25,000/\text{sec}$ peak.
- **Read QPS (Timeline):** $300,000 \text{ requests/sec}$ ($60:1$ read/write ratio).
- **Video Storage Ingest:** $50,000 \text{ hours uploaded/day} \times 10 \text{ GB/hour} = 500 \text{ TB/day}$ raw; $2.5 \text{ PB/day}$ post-transcoding (5 bitrate renditions).
- **Timeline Cache Memory:** 300M DAU $\times$ 800 cached post IDs $\times$ 16 bytes (8B post\_id + 8B timestamp) $\approx 3.84 \text{ TB RAM}$.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Edge API Gateways** | 50 $\times$ `c6i.4xlarge` (EKS) | 16 vCPU, 32 GB RAM, 12.5 Gbps | Terminates 300k QPS timeline reads, verifies JWT tokens, routes traffic |
| **Timeline Cache Cluster** | 40 $\times$ `r6i.2xlarge` (Redis Cluster) | 8 vCPU, 64 GB RAM per node | Sharded Redis storing user timelines in Sorted Sets, capped at 800 items/user |
| **Fan-Out Workers** | 60 $\times$ `c6i.2xlarge` (K8s HPA) | 8 vCPU, 16 GB RAM | Consumes Kafka `post.created` events, pushes post IDs into follower timelines |
| **Transcode GPU Cluster** | 80 $\times$ `g5.2xlarge` (Spot/On-Demand) | 1 NVIDIA A10G (24 GB VRAM), 8 vCPU | Hardware-accelerated NVENC encoding for 1080p, 720p, 480p HLS chunks |
| **Kafka Fan-Out Brokers** | 12 $\times$ `i3en.3xlarge` (KRaft) | 12 vCPU, 96 GB RAM, NVMe | Sustains massive fan-out messaging throughput without disk write bottlenecks |
| **Metadata & Feed DB** | 1 Primary + 3 Read Replicas | `db.r6i.8xlarge` (256 GB RAM) | Aurora PostgreSQL for user graphs, post metadata, and partitioned video assets |

#### Visual Architecture Blueprint
![Figure 17.4: Consumer Social Feed & Video Streaming Architecture](visuals/arch_social_video_platform.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Hybrid Fan-Out Feed Strategy (Celebrity Cutoff):**
   - **Regular Users (<10k followers) — Fan-out-on-write (Push):** When a regular user posts, asynchronous fan-out workers insert the `post_id` into every follower's timeline Redis Sorted Set (`ZADD timeline:{follower_id} {timestamp} {post_id}`). Reading the timeline requires a single $\mathcal{O}(1)$ `ZREVRANGEBYSCORE timeline:{user_id} +inf -inf LIMIT 0 20`.
   - **Celebrity Accounts ($\ge 10\text{k}$ followers) — Fan-out-on-read (Pull):** High-follower accounts do not fan out to millions of follower mailboxes. Instead, their posts append to a single `celebrity_outbox:{celebrity_id}` Sorted Set.
   - **Read-Time Dynamic Merge:** When a user requests their feed, the service fetches the user's personal timeline from Redis, queries the latest posts from all followed celebrities, and performs an in-memory $K$-way merge sort across the result lists in $< 15\text{ms}$.
2. **Adaptive Bitrate (ABR) Video Pipeline:**
   - **Direct-to-S3 Presigned Uploads:** Mobile/web clients request a presigned S3 PUT URL from the API gateway and stream binary MP4 chunks directly to S3. Gateway CPU and bandwidth remain untaxed.
   - **Asynchronous Transcode Farm:** S3 upload triggers an AWS SQS message. Autoscaled worker instances running FFmpeg with hardware acceleration (NVENC) slice videos into 6-second MPEG-TS/CMAF chunks across multiple resolutions: 1080p (6 Mbps), 720p (3 Mbps), 480p (1.5 Mbps), and 360p (800 kbps).
   - **Manifest Generation:** Produces HLS Master Playlist (`master.m3u8`) and sub-manifests pointing to chunk files (`chunk_001.ts`).
3. **Global Edge CDN & Origin Shielding:**
   - CloudFront / Fastly edge PoPs cache `.ts` video chunks aggressively ($99.2\%$ cache hit ratio).
   - S3 Origin Shield acts as a centralized caching tier, protecting the root S3 storage bucket from cache stampedes when a video goes viral.

#### API Contracts & Interface Specs
```json
// POST /v1/posts (Create Post with Media)
Header: Authorization: Bearer <token>
Header: X-Idempotency-Key: "usr42_post_20260901"
{
  "author_id": "usr_291a8f",
  "content_text": "Exploring distributed consensus under network partitions",
  "media_asset_ids": ["vid_a9128f01"],
  "visibility": "PUBLIC"
}
// GET /v1/timeline?user_id=usr_42&cursor=1728000000&limit=20
Response (200 OK):
{
  "posts": [
    {
      "post_id": "p_8812",
      "author": {"user_id": "usr_291a8f", "handle": "alex_tech", "is_verified": true},
      "content_text": "Exploring distributed consensus...",
      "video_manifest_url": "https://cdn.example.com/videos/vid_a9128f01/master.m3u8",
      "created_at": "2026-09-01T10:00:00Z"
    }
  ],
  "next_cursor": "1727985400"
}
```

#### Production Database Schema & Data Model (PostgreSQL DDL)
```sql
-- Partitioned posts table to sustain high write volumes
CREATE TABLE posts (
    post_id UUID NOT NULL,
    author_id UUID NOT NULL,
    content_text TEXT,
    media_manifest_url VARCHAR(512),
    like_count INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    visibility VARCHAR(10) NOT NULL DEFAULT 'PUBLIC',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (post_id, created_at)
) PARTITION BY RANGE (created_at);

-- Monthly partition tables
CREATE TABLE posts_2026_09 PARTITION OF posts
    FOR VALUES FROM ('2026-09-01 00:00:00+00') TO ('2026-10-01 00:00:00+00');

CREATE INDEX idx_posts_author ON posts(author_id, created_at DESC);

-- Video asset transcoding pipeline state
CREATE TABLE video_assets (
    asset_id UUID PRIMARY KEY,
    uploader_id UUID NOT NULL,
    raw_s3_key VARCHAR(512) NOT NULL,
    master_manifest_url VARCHAR(512),
    duration_seconds INT NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('UPLOADING', 'PROCESSING', 'READY', 'FAILED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_video_status ON video_assets(status) WHERE status = 'PROCESSING';
```

#### Step-by-Step Execution Sequence
1. **Post Ingest:** User publishes post. API Gateway validates payload and commits record into PostgreSQL `posts` table within an atomic transaction.
2. **Fan-Out Evaluation:** Fan-out coordinator checks author's follower count:
   - If $< 10,000$: Emits event to Kafka topic `fanout.push`. 60 worker pods read follower lists and pipe `ZADD timeline:{follower_id} {ts} {post_id}` in batches into Redis.
   - If $\ge 10,000$: Pushes `post_id` only to `celebrity_outbox:{author_id}`.
3. **Timeline Fetch & Merge:** Client queries `/v1/timeline`. Feed service executes Redis pipeline: fetches user's home timeline + pulls latest entries from followed celebrities' outboxes.
4. **K-Way Merge & Hydration:** In-memory min-heap merges posts into a strict reverse-chronological list of 20 items, hydrates text and author profiles from Redis cache, and returns response in $< 40\text{ms}$.
5. **Video Playback Initiation:** Client hits CDN for `master.m3u8`, measures network throughput, and selects optimal bitrate rendition (e.g., 720p). First video segment streams in $< 600\text{ms}$.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Celebrity Tweet Storm** | Mega-celebrity (100M followers) posts; naive push model spawns 100M Redis writes, causing 45-second latency spike. | **Strict Push/Pull Decoupling:** Accounts $\ge 10\text{k}$ followers are marked in metadata. Fan-out workers skip push entirely. Post is written to a single outbox key and pulled at query time by followers. |
| **Redis Timeline Cache Eviction Storm** | High memory usage triggers random Redis key eviction, wiping millions of active user feeds. | **Strict Bounded Deletion & Volatile-TTL:** Fan-out workers execute `ZREMRANGEBYRANK timeline:{id} 0 -801` after every insert, guaranteeing each user timeline never exceeds 800 items. Redis memory usage remains strictly bounded. |
| **Video Transcode Pod Hang / Crash** | FFmpeg hangs on malformed MP4 container, tying up expensive GPU worker resources. | **SQS Visibility Timeout with Dead Letter Queue:** Workers operate under strict 10-minute visibility timeouts with heartbeats. Process timeouts kill FFmpeg processes, while poisoned files move to DLQ after 3 failed attempts. |
| **Viral Video CDN Cache Stampede** | Trending breaking-news video experiences $100,000\text{ requests/sec}$ cache miss upon publication. | **Origin Shield & Request Coalescing:** CDN edge uses `proxy_cache_use_stale updating` and HTTP request collapsing. Only one HTTP GET reaches the S3 origin; all 99,999 other requests wait and serve from the cached chunk. |

#### Staff-Level Interview Verbalization
> *"To scale our social feed to 300 million daily users, we implement a Hybrid Push/Pull fan-out model bounded at 10,000 followers. Regular posts are pushed asynchronously into Redis Sorted Sets capped at 800 items per user, while celebrity posts are merged in memory at query time using a min-heap, completely eliminating write amplification storms. Video traffic completely bypasses the application server plane: clients upload directly to S3 via presigned URLs, SQS triggers GPU-accelerated FFmpeg transcode clusters to generate multi-bitrate HLS chunks, and CDN Origin Shielding protects storage infrastructure during viral traffic surges."*


### Solution 5: Consumer Scale — Real-Time Ride-Sharing Geospatial Dispatch System

#### Problem Statement & SLAs
Design a real-time ride-sharing dispatch system (Uber/Lyft).

- **Active Scale:** 10 million active drivers streaming GPS locations every 3 seconds.
- **Latency SLA:** Driver-rider matching $< 3\text{ seconds}$. Location update ingest $< 100\text{ms}$.
- **Marketplace SLA:** Zero double-dispatches (a single driver must never receive two conflicting ride assignments).

#### Capacity Estimation & Hardware Sizing
- **Location Ingest QPS:** $10,000,000 \text{ drivers} / 3 \text{ seconds} \approx 3.33 \text{ million QPS}$.
- **Network Ingress:** $3.33 \times 10^6 \times 64 \text{ bytes} \approx 213 \text{ MB/sec} = 1.7 \text{ Gbps}$.
- **Redis Spatial Index Memory:** $10,000,000 \text{ drivers} \times 96 \text{ bytes} \approx 960 \text{ MB RAM}$. Easily cached in-memory.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **WebSocket Gateway (WSS)** | 80 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Maintains 10M concurrent persistent TLS connections (~125k connections/pod) |
| **Kafka Location Ingest** | 16 $\times$ `i3en.2xlarge` (KRaft) | 8 vCPU, 64 GB RAM, NVMe | Ingests 3.33M QPS partitioned by H3 Resolution 6 parent cell |
| **Redis Spatial Cluster** | 24 $\times$ `r6i.2xlarge` (12 Primary + 12 Replica) | 8 vCPU, 64 GB RAM per node | Sharded by H3 parent cell; stores real-time driver coordinates via `GEOADD` |
| **Batch Dispatch Engine** | 32 $\times$ `c6i.4xlarge` (EKS Workers) | 16 vCPU, 32 GB RAM | Executes 5-second batch bipartite matching (Kuhn-Munkres / KD-tree) |
| **Trip Lifecycle DB** | 1 Primary + 3 Read Replicas | `db.r6i.8xlarge` (256 GB RAM) | Aurora PostgreSQL with PostGIS extension, monthly partitioned trip history |

#### Visual Architecture Blueprint
![Figure 17.5: Ride-Sharing Geospatial Dispatch System](visuals/arch_rideshare_geospatial.png){width=95%}

#### Architectural Workflow & Mechanics
1. **High-Throughput Telemetry Ingest:**
   - Mobile driver applications stream continuous GPS coordinates (Lat/Lon, Driver ID, Status) over persistent **Secure WebSockets (WSS)** into an edge load balancer cluster.
   - Updates stream into Kafka partitioned deterministically by high-level geographic region.
2. **Hierarchical Geospatial Indexing (Uber H3):**
   - Coordinates are indexed using **Uber H3 hexagonal grid cells**:
     - **Resolution 8 (~460m edge length, $0.74\text{ km}^2$ area):** Coarse spatial aggregation used for surge pricing supply/demand counters.
     - **Resolution 9 (~174m edge length, $0.10\text{ km}^2$ area):** Fine spatial resolution used for local driver candidate lookups.
   - Driver positions are cached in Redis Geospatial Sorted Sets: `GEOADD active_drivers:{h3_res8_parent} lon lat driver_id`.
3. **Hexagonal Spatial Neighbor Traversal ($k$-Ring Search):**
   - Unlike Geohash rectangles (which suffer from edge discontinuities where adjacent cells have completely different prefixes), H3 hexagons maintain uniform distance to all 6 contiguous neighbors.
   - A $k$-ring search (`h3.kRing(pickup_cell, k)`) expands outward symmetrically: $k=1$ yields 7 cells, $k=2$ yields 19 cells, and $k=3$ yields 37 cells, allowing seamless radius expansion across cell boundaries.
4. **Dynamic Surge Pricing Feedback Loop:**
   - Evaluates supply (available idle drivers) and demand (active ride requests) within each H3 Resolution 8 cell every 15 seconds:
     $$\text{Surge Multiplier} = \min\left(3.5, \max\left(1.0, 1.0 + \alpha \cdot \frac{\text{Unmatched Requests} - \text{Available Drivers}}{\text{Available Drivers} + \epsilon}\right)\right)$$

5. **Batch Matching vs. Greedy Dispatch:**
   - Rather than matching the first driver immediately (greedy local optimum), the engine pools ride requests into 5-second dispatch windows.
   - Runs bipartite matching to maximize global marketplace efficiency (minimizing average pickup ETA across all riders).

#### Haversine Great-Circle Distance Metric

To compute the spherical surface distance between rider coordinates $(\phi_1, \lambda_1)$ and driver coordinates $(\phi_2, \lambda_2)$ with earth radius $R \approx 6,371\text{ km}$:

$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$

Where $\Delta \phi = \phi_2 - \phi_1$ (latitude difference in radians) and $\Delta \lambda = \lambda_2 - \lambda_1$ (longitude difference in radians). Redis Geo internally computes this spherical distance via geohash integer bit-interleaving in $\mathcal{O}(1)$ time.

#### API Contracts & Interface Specs
```json
// POST /v1/trips/request (Rider requests a trip)
Header: Authorization: Bearer <token>
Header: X-Idempotency-Key: "trip_req_usr42_ts172800"
{
  "rider_id": "rdr_55812",
  "pickup": {"lat": 37.7749, "lon": -122.4194},
  "dropoff": {"lat": 37.3382, "lon": -121.8863},
  "ride_type": "STANDARD"
}
// Response (201 Created):
{
  "trip_id": "trip_a91f2",
  "status": "SEARCHING",
  "surge_multiplier": 1.4,
  "estimated_fare_cents": 3250,
  "pickup_h3_res9": 617700169958293503,
  "eta_seconds": 180
}
```

#### Production Database Schema & Data Model (PostgreSQL DDL)
```sql
CREATE EXTENSION IF NOT EXISTS postgis;

-- Partitioned trip history database
CREATE TABLE trips (
    trip_id UUID NOT NULL,
    rider_id UUID NOT NULL,
    driver_id UUID,
    pickup_geom GEOMETRY(Point, 4326) NOT NULL,
    dropoff_geom GEOMETRY(Point, 4326) NOT NULL,
    pickup_h3_res8 BIGINT NOT NULL,
    pickup_h3_res9 BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('REQUESTED','MATCHED','IN_PROGRESS','COMPLETED','CANCELLED')),
    surge_multiplier DECIMAL(3,1) NOT NULL DEFAULT 1.0,
    fare_cents INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trip_id, created_at)
) PARTITION BY RANGE (created_at);

-- Monthly partition
CREATE TABLE trips_2026_09 PARTITION OF trips
    FOR VALUES FROM ('2026-09-01 00:00:00+00') TO ('2026-10-01 00:00:00+00');

CREATE INDEX idx_trips_driver ON trips(driver_id, created_at DESC);
CREATE INDEX idx_trips_rider ON trips(rider_id, created_at DESC);
CREATE INDEX idx_trips_h3_res8 ON trips(pickup_h3_res8);
CREATE INDEX idx_trips_pickup_spatial ON trips USING GIST(pickup_geom);
```

#### Step-by-Step Execution Sequence
1. **Telemetry Streaming:** Driver app transmits `(driver_id, lat, lon, status)` every 3 seconds over WSS. Gateway forwards payload to Kafka topic `driver.locations`.
2. **H3 Index Update:** Location workers decode GPS coordinates, compute H3 index, and issue `GEOADD active_drivers:{h3_res8} lon lat driver_id` in Redis with 10-second TTL.
3. **Trip Request & Surge Calculation:** Rider submits trip request. Dispatch service calculates real-time surge multiplier using the ratio of unmatched requests to available drivers in the pickup H3 Resolution 8 cell.
4. **Candidate Gathering ($k$-Ring Expansion):** Dispatch Engine queries Redis Geo for nearest drivers within 3km radius using `GEOSEARCH active_drivers:{h3_res8} FROMLONLAT lon lat BYRADIUS 3 km ASC`.
5. **Atomic Assignment & WebSocket Offer:** Dispatcher acquires atomic lock on driver (`SETNX match:driver:{id} trip_id EX 15`). If acquired, sends trip offer to driver over WebSocket. If driver accepts, state transitions to `MATCHED` in PostgreSQL.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **WebSocket Gateway Node Crash** | Pod failure severs 125,000 persistent driver connections simultaneously. | **Exponential Reconnect Jitter:** Client mobile SDKs detect socket drop and reconnect with full randomized exponential backoff ($1\text{--}15\text{s}$) to prevent thundering herd crashes on surviving gateway pods. |
| **Concurrent Double Dispatch** | Two riders near the same driver are assigned simultaneously by concurrent dispatch workers. | **Atomic Redis Lock with Lease:** Matching engine uses Redis atomic conditional write: `SET lock:driver:{id} {trip_id} NX EX 15`. Second worker fails lock acquisition and immediately evaluates the next nearest driver. |
| **Ghost / Zombie Drivers** | Driver phone enters tunnel or battery dies; outdated location stays in cache, leading to ghost match. | **Strict 10-Second TTL on Redis Keys:** Redis spatial entries expire automatically after 10 seconds. If a driver fails to heartbeat for 2 cycles (6 seconds), they are automatically purged from active dispatch candidates. |
| **Flash Mob Surge Spike** | Massive concert ends; 20,000 riders request rides simultaneously, causing volatile surge price swings. | **Surge Dampening & Moving Average:** Surge pricing formula applies exponential moving average (EMA) smoothing over a 3-minute window and caps maximum surge multiplier change at $+0.2\times$ per minute. |

#### Staff-Level Interview Verbalization
> *"Our ride-sharing dispatch system partitions 3.3 million QPS of GPS telemetry using Uber H3 hexagonal indexing. We shard Redis Geo clusters by H3 Resolution 8 cells, enabling sub-millisecond $k$-ring neighbor queries that eliminate Geohash rectangular edge artifacts. Rather than naive greedy assignment, our dispatch engine batches trip requests into 5-second windows to optimize global pickup ETAs, enforcing atomic Redis reservation locks to strictly prevent double-dispatch races under concurrent load."*


### Solution 6: Modern AI/ML — Distributed Vector Search & RAG Knowledge Engine

#### Problem Statement & SLAs
Design an enterprise Retrieval-Augmented Generation (RAG) knowledge search system over millions of unstructured documents.

- **Document Corpus:** 100 million document chunks (average 512 tokens per chunk).
- **Latency SLA:** Hybrid vector search $p99 < 50\text{ms}$. End-to-end LLM generation $p99 < 2\text{s}$ (time-to-first-token $< 400\text{ms}$).
- **Precision SLA:** Zero hallucinated internal citations; $100\%$ source traceability.

#### Capacity Estimation & Hardware Sizing
- **Embedding Dimensions:** 768-dimensional float32 vectors ($768 \times 4 \text{ bytes} = 3,072 \text{ bytes/vector}$).
- **Raw Vector Storage:** $100,000,000 \text{ vectors} \times 3,072 \text{ bytes} \approx 307.2 \text{ GB RAM}$.
- **HNSW Graph Overhead:** Hierarchical graph edges add $\approx 40\%$ RAM overhead ($M=16, efConstruction=200$), totaling $\approx 430 \text{ GB RAM}$ for in-memory graph traversal.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Vector DB Cluster** | 8 $\times$ `r6i.4xlarge` (Milvus / Qdrant) | 16 vCPU, 128 GB RAM per node | Sharded in-memory HNSW vector index (~1 TB total RAM capacity) |
| **GPU Embedding Inference** | 12 $\times$ `g5.xlarge` (Triton Server) | 1 NVIDIA A10G (24 GB VRAM), 4 vCPU | Serves BGE-large-en embedding models at 6,000 chunks/sec throughput |
| **Sparse Keyword Search** | 6 $\times$ `r6i.2xlarge` (OpenSearch) | 8 vCPU, 64 GB RAM | BM25 inverted index for exact identifier, error code, and keyword matches |
| **RAG Orchestrator & Cache** | 16 $\times$ `c6i.2xlarge` (EKS Pods) | 8 vCPU, 16 GB RAM | Evaluates semantic cache, executes hybrid search, coordinates LLM stream |
| **Metadata & Chunk Store** | 1 Primary + 1 Replica | `db.r6i.4xlarge` (128 GB RAM) | Aurora PostgreSQL for chunk text, parent doc metadata, and lineage |

#### Visual Architecture Blueprint
![Figure 17.6: Distributed Vector Search & RAG Architecture](visuals/arch_vector_rag_system.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Document Ingestion & Chunking Pipeline:**
   - Ingestion workers scrape enterprise documents (PDFs, Markdown, Confluence, Jira), strip boilerplate, and apply recursive chunking (512 tokens with 64-token sliding window overlap).
   - Ingestion pipeline computes cryptographic hash per chunk (`chunk_hash`) to ensure idempotent re-indexing.
   - GPU worker clusters (running NVIDIA Triton Inference Server) compute 768-dimensional dense embeddings.
2. **Dual Index Storage Architecture:**
   - **Dense Vectors:** High-dimensional embeddings are indexed using **HNSW (Hierarchical Navigable Small World)** graphs in vector databases (Milvus / Qdrant). HNSW delivers sub-10ms approximate nearest neighbor (ANN) search with $\ge 98\%$ recall.
   - **Sparse Lexical Keywords:** Raw text chunks are tokenized and stored in **BM25 / OpenSearch** indexes, ensuring exact keyword matches (such as function names, error codes, and IDs) are never missed by dense semantic vectors.
3. **Semantic Caching Layer:**
   - Before executing vector search, the orchestrator queries a Redis Semantic Cache containing past queries and generated responses.
   - If the cosine similarity between the incoming query vector and a cached query vector $\ge 0.96$, the cached answer is returned immediately ($5\text{ms}$ latency, zero LLM cost).
4. **Hybrid Retrieval & Reciprocal Rank Fusion (RRF):**
   - User queries execute simultaneous dense ANN vector similarity search ($\mathcal{O}(\log N)$) and sparse BM25 keyword matching.
   - Results are unified and reranked using Reciprocal Rank Fusion:
     $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k = 60)$$

   - The top 50 candidates pass through a cross-encoder reranker (e.g., BGE-reranker-large), extracting the top 5 most relevant context snippets.
5. **Context Assembly & LLM Generation:**
   - Chunks are injected into a structured prompt template along with strict system instructions ("Answer only based on the provided context; cite chunk IDs for every claim").
   - The prompt streams to an LLM (Claude / GPT-4 / Llama 3) via Server-Sent Events (SSE).

#### API Contracts & Interface Specs
```json
// POST /v1/search (Hybrid RAG Query)
Header: Authorization: Bearer <token>
Header: X-Correlation-ID: "rag_req_99812a"
{
  "query": "How does circuit breaker pattern prevent cascade failures?",
  "top_k": 5,
  "enable_semantic_cache": true,
  "stream": false
}
// Response (200 OK):
{
  "query_id": "qry_8812f",
  "cached": false,
  "retrieval_latency_ms": 38,
  "chunks": [
    {
      "chunk_id": "chk_9a12",
      "score": 0.934,
      "document_title": "Distributed Resiliency Patterns",
      "text": "The circuit breaker transitions between CLOSED, OPEN, and HALF-OPEN..."
    }
  ],
  "generated_answer": "Circuit breakers prevent cascade failures by tripping to an OPEN state...",
  "citations": ["chk_9a12"],
  "model": "llama-3-70b-instruct",
  "total_latency_ms": 820
}
```

#### Production Database Schema & Data Model (PostgreSQL + pgvector DDL)
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    chunk_index INT NOT NULL,
    chunk_hash VARCHAR(64) NOT NULL,
    content_text TEXT NOT NULL,
    token_count INT NOT NULL,
    embedding vector(768) NOT NULL,
    embedding_model VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Fast cosine similarity search index via HNSW
CREATE INDEX idx_chunks_hnsw_cosine ON document_chunks 
    USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 200);

CREATE INDEX idx_chunks_doc ON document_chunks(document_id, chunk_index);

-- Semantic query cache table
CREATE TABLE semantic_query_cache (
    query_id UUID PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_vector vector(768) NOT NULL,
    cached_response TEXT NOT NULL,
    hit_count INT DEFAULT 1,
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_query_cache_vector ON semantic_query_cache 
    USING hnsw (query_vector vector_cosine_ops);
```

#### Step-by-Step Execution Sequence
1. **Query Embedding:** Orchestrator sends query text to Triton GPU cluster; receives 768d float32 vector within 6ms.
2. **Semantic Cache Check:** Orchestrator queries Redis/pgvector semantic cache. If cosine similarity $\ge 0.96$, returns cached answer directly.
3. **Parallel Hybrid Search:** Orchestrator dispatches parallel search requests:
   - Dense ANN search against Milvus HNSW cluster ($k=50$).
   - Sparse lexical BM25 query against OpenSearch cluster ($k=50$).
4. **Reciprocal Rank Fusion & Reranking:** Consolidates 100 candidate chunks using RRF ($k=60$). Passes top 25 candidates to cross-encoder reranking model on GPU, selecting top 5 context snippets.
5. **Prompt Synthesis & LLM Streaming:** Orchestrator injects top 5 chunks into context window, verifies token budget, and streams response tokens to client via Server-Sent Events (SSE).

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Vector DB Memory Exhaustion** | Ingestion of 50M new documents exceeds cluster RAM, causing OOM kills and search downtime. | **IVF-PQ Scalar Quantization Fallback:** Vector DB transitions historical collections from HNSW to Product Quantization (IVF-PQ8). Quantizing 32-bit floats to 8-bit codes reduces memory footprint by $4\times$ ($75\%$ reduction) with $< 2\%$ recall degradation. |
| **LLM Provider Outage / 429 Rate Limits** | Primary LLM API returns HTTP 429 or 503 during peak business hours, halting all query generation. | **Multi-Model Fallback Gateway:** Orchestrator implements circuit-breaker fallback: primary provider (Claude 3.5 Sonnet) $\rightarrow$ secondary provider (GPT-4o) $\rightarrow$ self-hosted fallback model (vLLM Llama-3-70B on GPU cluster). |
| **Context Window Hallucination** | Irrelevant chunks fill context window, causing LLM to fabricate non-existent facts. | **Cross-Encoder Relevance Threshold:** Any chunk with cross-encoder score $< 0.40$ is strictly discarded before prompt assembly. If zero chunks pass, system responds: *"Insufficient internal documentation to answer this question."* |
| **Stale Chunk Desynchronization** | Document is updated in source CMS, but obsolete chunk remains in vector index, serving contradictory answers. | **CDC Tombstone Invalidation:** Postgres Debezium CDC pipeline tails CMS mutations. Any document update triggers immediate tombstone deletion of old `document_id` chunk IDs from both Milvus and OpenSearch before re-indexing. |

#### Staff-Level Interview Verbalization
> *"Our enterprise RAG architecture overcomes the limitations of pure vector search by implementing a dual-retrieval pipeline: dense HNSW approximate nearest neighbors for semantic intent, and sparse BM25 for exact token and code matching. Results are merged via Reciprocal Rank Fusion and refined by a GPU cross-encoder reranker. We achieve sub-50ms retrieval while protecting LLM operating costs using an in-memory semantic cache with a 0.96 cosine threshold, backed by multi-provider circuit-breaking fallbacks."*


### Solution 7: Cloud Storage — Distributed File Storage & Sync Engine (Google Drive / Dropbox)

#### Problem Statement & SLAs
Design a distributed file storage and sync platform capable of handling multi-gigabyte files across millions of devices.

- **Scale:** 500 million registered users, 100 million active files synced per day.
- **Latency SLA:** File metadata sync $< 200\text{ms}$. Delta upload latency proportional to modified byte count only.
- **Consistency SLA:** Strict block immutability and file version ordering (Vector Clocks). Zero data loss ($99.999999999\%$ 11-nines durability).

#### Capacity Estimation & Hardware Sizing
- **Average File Size:** 2 MB average. Daily storage ingest: $100 \times 10^6 \times 2 \text{ MB} = 200 \text{ TB/day}$.
- **Block Size (Chunking):** 4 MB average variable chunk size via Rabin Fingerprinting.
- **Metadata Storage:** 1 billion files $\times 1 \text{ KB metadata} = 1 \text{ TB}$ metadata DB index in CockroachDB/PostgreSQL.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Sync WebSocket Gateways** | 40 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Maintains persistent client sync sockets, pushes manifest updates |
| **Chunk Upload Gateways** | 30 $\times$ `c6i.xlarge` (EKS) | 4 vCPU, 8 GB RAM | Evaluates SHA-256 chunk hashes, issues presigned S3 PUT URLs |
| **Hash Deduplication Cache** | 8 $\times$ `r6i.xlarge` (Redis Cluster) | 4 vCPU, 32 GB RAM per node | Caches active chunk SHA-256 hashes to bypass database deduplication queries |
| **Distributed Metadata DB** | 6 $\times$ `r6i.4xlarge` (CockroachDB) | 16 vCPU, 128 GB RAM | Distributed ACID transactions for file trees, revisions, and vector clocks |
| **Block Storage (CAS)** | AWS S3 Standard + Glacier | Multi-AZ Durability | Content-addressable storage bucket for immutable 4MB encrypted chunks |

#### Visual Architecture Blueprint
![Figure 17.7: Distributed File Storage & Sync Engine Architecture](visuals/arch_drive_sync_storage.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Client-Side Content-Defined Chunking (Rabin Fingerprinting):**
   - Fixed-size chunking (e.g., rigid 4MB blocks) suffers from boundary shifting: inserting a single byte at offset 0 shifts every downstream boundary, requiring a 100% re-upload of a 10GB file.
   - **Rabin Fingerprinting CDC** slides a rolling polynomial hash window over byte stream $(b_1, \dots, b_k)$:
     $$H(b_1, \dots, b_k) = \left(\sum_{i=1}^k b_i \cdot p^{k-i}\right) \pmod M$$

   - When rolling hash $H \equiv 0 \pmod D$ (where $D = 2^{22} = 4\text{ MB}$), a chunk boundary is declared. Byte insertions only alter the immediate local chunk boundary; all subsequent chunks maintain identical boundary hashes, reducing sync bandwidth by $> 99\%$.
2. **Content-Addressable Storage (CAS) & Global Deduplication:**
   - Every chunk is identified by its cryptographic digest: `SHA-256(chunk_bytes)`.
   - Before uploading, the client transmits the list of chunk hashes to the Deduplication Gateway.
   - If a chunk hash exists in the global `file_blocks` index (uploaded by any user), the server simply increments `reference_count` and skips data upload.
3. **Optimistic Sync & Vector Clock Conflict Resolution:**
   - Concurrent edits across offline devices are tracked via **Vector Clocks**:
     $$V = [c_1, c_2, \dots, c_n]$$

   - If client $A$'s vector dominates client $B$'s ($V_A > V_B$), the update is cleanly fast-forwarded.
   - If vectors conflict ($V_A \parallel V_B$), the sync engine creates a non-destructive conflict fork (e.g., `Project_Spec (Conflicted copy from MacBook).docx`), preserving both revisions without data loss.

#### API Contracts & Interface Specs
```json
// POST /v1/files/check_chunks (Deduplication Query)
Header: Authorization: Bearer <token>
{
  "file_path": "/Documents/architecture_v2.pdf",
  "total_bytes": 12582912,
  "chunk_hashes": [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
    "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d"
  ]
}
// Response (200 OK):
{
  "existing_chunks": [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ],
  "missing_chunks": [
    {
      "chunk_hash": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
      "upload_presigned_url": "https://s3.amazonaws.com/storage/chunk_4b22?AWSAccessKeyId=..."
    },
    {
      "chunk_hash": "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
      "upload_presigned_url": "https://s3.amazonaws.com/storage/chunk_ef2d?AWSAccessKeyId=..."
    }
  ]
}
```

#### Production Database Schema & Data Model (CockroachDB / PostgreSQL DDL)
```sql
-- Global deduplicated content-addressable block registry
CREATE TABLE file_blocks (
    block_hash VARCHAR(64) PRIMARY KEY, -- SHA-256 hex
    storage_url VARCHAR(512) NOT NULL,
    byte_size INT NOT NULL,
    reference_count INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- File metadata and version manifests
CREATE TABLE file_manifests (
    file_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    vector_clock JSONB NOT NULL DEFAULT '{}',
    chunk_hashes JSONB NOT NULL, -- Ordered array of SHA-256 block hashes
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_path UNIQUE (user_id, file_path)
);

CREATE TABLE sync_devices (
    device_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    device_name VARCHAR(128) NOT NULL,
    last_synced_version INT NOT NULL,
    last_heartbeat TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_manifests_user ON file_manifests(user_id);
CREATE INDEX idx_devices_user ON sync_devices(user_id);
```

#### Step-by-Step Execution Sequence
1. **Local Change Detection:** Background OS file watcher detects file mutation, slices bytes using Rabin Fingerprinting, and generates SHA-256 chunk checksums.
2. **Global Deduplication Check:** Client queries `/v1/files/check_chunks`. Redis/DB cache identifies chunks already stored globally, skipping 80% of byte transfers.
3. **Resumable Parallel S3 Upload:** Client uploads only missing chunks directly to S3 via presigned PUT URLs with S3 multi-part verification.
4. **Atomic Manifest Commit:** Client submits atomic manifest update with updated vector clock (`{"client_A": 4}`). CockroachDB commits metadata transaction.
5. **Real-Time Peer Notification:** Sync service emits notification over persistent WebSockets to user's other registered devices (`laptop`, `mobile`), triggering delta pull.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Network Drop During 5GB Upload** | Client connection drops at 95% upload progress; naive systems force complete restart. | **Chunked Resumable Uploads with S3 Multipart:** Each 4MB chunk is an independent S3 PUT with MD5/SHA checksum. On reconnect, client re-queries missing chunks and resumes only remaining blocks. |
| **Concurrent Offline Edit Collision** | User edits document on laptop while offline on a flight; spouse edits same file on desktop. | **Non-Destructive Vector Clock Forking:** Manifest commit detects concurrent vector clock branch ($V_A \parallel V_B$). Server accepts both commits, renaming second file to `File_Name (Conflicted Copy).ext`. |
| **SHA-256 Hash Collision** | Two different user files produce identical SHA-256 hash, causing data corruption. | **Length Verification & Secondary Salt:** Probability of 256-bit hash collision is $2^{-128} \approx 10^{-38}$. To guarantee zero data corruption, server verifies both SHA-256 hash AND exact byte size before deduplication match. |
| **Orphaned Chunk Leakage** | User deletes file, but referenced chunks remain in S3 indefinitely, accumulating storage costs. | **Mark-and-Sweep Garbage Collection:** File deletion decrements `reference_count` in `file_blocks`. Async worker purges S3 blocks where `reference_count <= 0` after a 30-day grace retention period. |

#### Staff-Level Interview Verbalization
> *"Our cloud file storage engine achieves extreme bandwidth efficiency by combining Content-Defined Chunking via Rabin Fingerprints with global Content-Addressable Storage. By decoupling immutable 4MB block uploads to S3 from metadata commits in CockroachDB, we ensure that inserting a single byte in a 10GB file transfers only the modified chunk. Concurrent multi-device edits are tracked via Vector Clocks, guaranteeing non-destructive branch resolution without data loss during offline split-brain scenarios."*


### Solution 8: Search Engine — Distributed Web Crawler & Search Indexer (Google Search)

#### Problem Statement & SLAs
Design a distributed web crawler and search indexer capable of crawling billions of web pages and updating an inverted search index.

- **Scale:** 10 billion web pages crawled per month ($\approx 3,850 \text{ pages/sec}$ sustained, $15,000/\text{sec}$ peak).
- **Latency SLA:** Search query execution $p99 < 100\text{ms}$ over a 50-billion document corpus.
- **Politeness SLA:** Enforce robots.txt and strict per-host rate limits (no more than 1 request/sec per target domain).

#### Capacity Estimation & Hardware Sizing
- **Page Size:** 100 KB average HTML page.
- **Storage Ingest:** $3,850 \text{ pages/sec} \times 100 \text{ KB} = 385 \text{ MB/sec} = 3.08 \text{ Gbps}$ ingress.
- **Monthly Storage:** $10 \times 10^9 \text{ pages} \times 100 \text{ KB} = 1 \text{ PB/month}$ raw HTML; compressed WARC $\approx 250 \text{ TB/month}$.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Crawler Fetcher Pods** | 100 $\times$ `c6i.xlarge` (EKS) | 4 vCPU, 8 GB RAM, 12.5 Gbps | Asynchronous non-blocking HTTP fetchers with `libcurl` / epoll event loop |
| **URL Frontier Cluster** | 12 $\times$ `r6i.2xlarge` (Redis / RocksDB) | 8 vCPU, 64 GB RAM per node | Two-tier priority & politeness queues, in-memory Bloom filter deduplication |
| **Bigtable / Raw HTML Store**| 30 $\times$ `i3en.3xlarge` (HBase/Bigtable) | 12 vCPU, 96 GB RAM, NVMe | 7.5 TB NVMe SSD per node; stores raw compressed document payloads |
| **Inverted Index Cluster** | 24 $\times$ `r6i.4xlarge` (OpenSearch) | 16 vCPU, 128 GB RAM | Columnar posting lists compressed via Variable-Byte and Elias-Fano delta encoding |
| **PageRank Spark Cluster** | 20 $\times$ `r6i.4xlarge` (Spot EMR) | 16 vCPU, 128 GB RAM | Nightly distributed graph power-iteration computing global PageRank authority |

#### Visual Architecture Blueprint
![Figure 17.8: Distributed Web Crawler & Inverted Search Indexer Architecture](visuals/arch_web_crawler_search.png){width=95%}

#### Architectural Workflow & Mechanics
1. **URL Frontier & Politeness Scheduling:**
   - The URL Frontier prevents self-inflicted DDoS by separating scheduling into two distinct queue tiers:
     - **Priority Queues (F1):** Categorizes URLs based on PageRank authority and update frequency.
     - **Politeness Queues (F2):** Maps each target hostname (e.g., `wikipedia.org`) to an isolated FIFO queue. A min-heap scheduler ensures that no two requests to the same hostname fire within a 1-second politeness window.
   - An asynchronous local DNS resolver (`c-ares`) caches DNS records in memory with 24-hour TTLs, eliminating DNS round-trip bottlenecks.
2. **HTML Parsing & Near-Duplicate Filtering (SimHash Algorithm):**
   - Fetched documents are parsed to extract outgoing hyperlinks (fed back into the frontier) and clean textual content.
   - Computes a **64-bit SimHash fingerprint** per document:
     $$V[i] = \sum_{w \in \text{Doc}} \text{weight}(w) \times \begin{cases} +1 & \text{if } \text{hash}(w)_i = 1 \\ -1 & \text{if } \text{hash}(w)_i = 0 \end{cases}$$

   - Final SimHash bit $i = 1$ if $V[i] > 0$, else $0$. Two documents are near-duplicates if their **Hamming Distance $\le 3$ bits** (calculated via bitwise XOR and `popcount`), pruning $>90\%$ of duplicate web pages (mirrors, syndications, scrape spam).
3. **Inverted Index Construction & Posting List Compression:**
   - Tokenizes text into inverted posting lists mapping terms to occurrences and token offsets:
     $$\text{"algorithm"} \rightarrow [(\text{Doc1}, [14, 88]), (\text{Doc8}, [3]), (\text{Doc104}, [201])]$$

   - Uses **Variable-Byte (VByte) delta encoding**: instead of storing absolute 32-bit doc IDs, stores gap deltas ($\Delta_i = \text{docID}_i - \text{docID}_{i-1}$), compressing integer storage from 32 bits down to an average of 6 bits per entry.
4. **PageRank & Graph Scoring:**
   - Hyperlink structures are written to a distributed graph database. Distributed graph algorithms compute global PageRank scores, which are joined with inverted indexes during query execution.

#### API Contracts & Interface Specs
```json
// GET /v1/search?q=distributed+consensus&limit=10&search_after=d_10482
Response (200 OK):
{
  "results": [
    {
      "doc_id": "d_10482",
      "title": "Raft Consensus Algorithm Explained",
      "url": "https://example.com/raft",
      "snippet": "Raft achieves consensus via leader election and replicated state...",
      "pagerank_score": 0.00147,
      "last_crawled_at": "2026-08-30T14:22:10Z"
    }
  ],
  "total_results": 248100,
  "next_cursor": "d_10483",
  "query_latency_ms": 38
}
```

#### Production Database Schema & Data Model (Bigtable + PostgreSQL DDL)
```sql
-- Crawled Pages Metadata (PostgreSQL / CockroachDB)
CREATE TABLE crawled_pages (
    doc_id UUID PRIMARY KEY,
    url VARCHAR(2048) NOT NULL,
    domain_host VARCHAR(255) NOT NULL,
    simhash BIGINT NOT NULL,
    http_status INT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    pagerank_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    last_crawled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_crawled_url UNIQUE (url)
);

CREATE TABLE frontier_domains (
    domain_host VARCHAR(255) PRIMARY KEY,
    crawl_delay_ms INT NOT NULL DEFAULT 1000,
    last_fetched_at TIMESTAMPTZ NOT NULL,
    robots_txt TEXT,
    robots_expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_pages_simhash ON crawled_pages(simhash);
CREATE INDEX idx_pages_domain ON crawled_pages(domain_host, last_crawled_at);
```

#### Step-by-Step Execution Sequence
1. **URL Enqueue & Deduplication:** Frontier receives URL candidates, checks Redis Bloom Filter ($10\text{B keys}, k=10$). If bit is 0, adds URL to Bloom filter and pushes to domain FIFO politeness queue.
2. **Politeness Dequeue & Fetch:** Worker pops URL whose domain has satisfied the 1-second delay. Asynchronous fetcher resolves IP via local DNS cache, downloads HTML stream, and enforces a strict 10MB payload ceiling.
3. **Parsing & SimHash Check:** Parser extracts outgoing URLs, normalizes relative links, and computes 64-bit SimHash. If Hamming distance to existing documents $\le 3$, skips downstream indexing.
4. **Posting List Indexing:** Text tokenizer updates inverted index segments in OpenSearch, compressing document gap deltas via VByte encoding.
5. **Batch PageRank Iteration:** Graph builder updates web link topology in Spark EMR, recalculating PageRank scores across 50 billion nodes.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Spider Trap / Infinite Calendar Loop** | Malicious or poorly designed site generates infinite unique URLs (`/events?date=2026-09-01`), draining crawl budget. | **Max Depth & URL Path Pattern Limits:** URL frontier strictly caps crawl path depth at 16 levels, enforces max 500 URLs per subdomain, and applies regex heuristic guards against repeating path segments. |
| **Target Host Overload / DDoS Accusation** | 50 crawler pods simultaneously hit a small blog, knocking it offline. | **Centralized Host-Level Rate Locking:** All fetch requests must acquire a Redis token from `token_bucket:{domain_host}` before initiating HTTP connection. Politeness delay is strictly capped at $\ge 1\text{s}$. |
| **DNS Resolver Bottleneck** | 15,000 HTTP requests/sec cause outbound UDP DNS request drops, stalling workers. | **Local Asynchronous DNS Caching:** Workers use non-blocking `c-ares` library with a local `dnsmasq` cache per pod, configuring 24-hour TTLs to bypass external root DNS servers. |
| **Tar-Bomb / Infinite Byte Stream Attack** | Web server streams an infinite random byte generator (`/dev/urandom`), exhausting pod memory. | **Streaming Chunk Threshold Termination:** HTTP client inspects `Content-Length` header; if $> 10\text{ MB}$, connection is terminated immediately. A streaming byte counter forcibly severs socket if downloaded bytes exceed 10 MB. |

#### Staff-Level Interview Verbalization
> *"Our web crawler architecture resolves the dual challenge of throughput and politeness using a two-tier URL Frontier. Priority queues order crawling by PageRank authority, while host-level politeness queues enforce 1-second delays via in-memory Redis token buckets. We eliminate duplicate processing across billions of pages using a 64-bit SimHash near-duplicate detector with bitwise Hamming distance $\le 3$, compressing inverted index posting lists via Variable-Byte delta encoding to serve search queries under 100 milliseconds."*


### Solution 9: Real-Time Chat — Distributed Messaging & Presence Platform (WhatsApp / Slack / Discord)

#### Problem Statement & SLAs
Design a real-time messaging and user presence platform supporting 1-on-1 and group chats.

- **Scale:** 500 million daily active users (DAU), 50 billion messages/day ($\approx 580,000 \text{ msg/sec}$ average, $\approx 1.5 \text{ million msg/sec}$ peak).
- **Latency SLA:** End-to-end message delivery $p99 < 100\text{ms}$.
- **Presence SLA:** Online/Offline state propagation $< 2\text{ seconds}$.
- **Ordering SLA:** Strict monotonic message sequencing per channel.

#### Capacity Estimation & Hardware Sizing
- **Message Bandwidth:** $580,000 \text{ msg/sec} \times 500 \text{ bytes} = 290 \text{ MB/sec} = 2.32 \text{ Gbps}$ sustained; $6.0 \text{ Gbps}$ peak.
- **Storage:** $50 \times 10^9 \text{ msgs/day} \times 500 \text{ bytes} = 25 \text{ TB/day}$ in Cassandra/ScyllaDB; $750 \text{ TB/month}$.
- **Active Connections:** 18 million concurrent online users.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **WebSocket Gateway (WSS)** | 120 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Linux epoll / Netty non-blocking I/O; ~150k persistent TLS sockets per node |
| **Message Store (ScyllaDB)**| 30 $\times$ `i3en.3xlarge` (Wide-Column) | 12 vCPU, 96 GB RAM, NVMe | 7.5 TB NVMe SSD per node; partitioned by channel ID and monthly bucket |
| **Presence Cluster** | 16 $\times$ `r6i.2xlarge` (Redis Cluster) | 8 vCPU, 64 GB RAM per node | Tracks online status using Redis Bitmaps (500M users in ~60 MB RAM) |
| **Kafka Fan-Out Brokers** | 16 $\times$ `i3en.2xlarge` (KRaft) | 8 vCPU, 64 GB RAM, NVMe | High-throughput pub/sub delivering messages to active session gateways |
| **Mobile Push Dispatchers** | 24 $\times$ `c6i.xlarge` (EKS Workers) | 4 vCPU, 8 GB RAM | HTTP/2 multiplexed dispatch to Apple APNs and Google FCM v1 |

#### Visual Architecture Blueprint
![Figure 17.9: Real-Time Messaging & Presence Platform Architecture](visuals/arch_chat_messaging_presence.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Stateful Connection Management (Linux C10M Epoll):**
   - Edge **WebSocket Gateway Clusters** maintain millions of long-lived, persistent TLS connections from web and mobile clients using non-blocking Netty event loops (`SO_REUSEPORT`, TCP keepalive).
2. **User Presence Engine & Bitmap Optimization:**
   - Instead of allocating bulky JSON records per user, presence tracks heartbeats via **Redis Bitmaps**:
     - Users are assigned a 32-bit integer ID.
     - When user #104,821 sends a heartbeat, Redis executes `SETBIT presence:active 104821 1` with a 45-second sliding TTL.
     - Querying presence for an entire friend list requires a single atomic `BITOP AND` or pipelined `GETBIT` calls, mapping 500 million users in just **60 MB of RAM**.
3. **Message Persistence & Partition Bucketing:**
   - Cassandra partitions that exceed 100 MB suffer from severe JVM garbage collection pauses during compaction.
   - The message store partitions chat logs by a composite key: `((channel_id, bucket_month), message_id)`.
   - Ordering is guaranteed within each channel using monotonic **TIMEUUID** clustering keys.
4. **Group Chat Fan-Out Decoupling:**
   - **Small Groups (<100 members):** Message service resolves active WebSocket gateway node for each recipient via Redis session map and pushes payload over gRPC.
   - **Mega-Channels (>1,000 members):** Messages are not fanned out immediately to inactive users. Instead, active connected members receive the message via Redis Pub/Sub, while dormant users fetch deltas on demand when they open the channel.
5. **End-to-End Encryption (Signal Double Ratchet):**
   - Cryptographic ratchet derives ephemeral symmetric keys per message. The server acts as a zero-knowledge untrusted relay, persisting only encrypted ciphertext blobs.

#### API Contracts & Interface Specs
```json
// POST /v1/messages/send
Header: Authorization: Bearer <token>
Header: X-Idempotency-Key: "cm_f81d4fae_usr88102"
{
  "channel_id": "ch_grp_42a1",
  "client_message_id": "cm_f81d4fae",
  "encrypted_content": "A1b8...<base64-ciphertext>...",
  "media_url": null,
  "client_timestamp_ms": 1724601600120
}
// GET /v1/messages?channel_id=ch_grp_42a1&bucket_id=202609&before_id=7e2a91f0-5ef7-11eb-ae93-0242ac130002&limit=50
Response (200 OK):
{
  "messages": [
    {
      "message_id": "7e2a91f0-5ef7-11eb-ae93-0242ac130002",
      "sender_id": "usr_88102",
      "encrypted_content": "A1b8...",
      "sent_at": "2026-09-01T10:00:00.120Z"
    }
  ],
  "has_more": true
}
```

#### Production Database Schema & Data Model (Cassandra CQL + PostgreSQL DDL)
```sql
-- ScyllaDB / Cassandra Message Store
CREATE KEYSPACE chat_system 
    WITH replication = {'class': 'NetworkTopologyStrategy', 'us-east': 3, 'us-west': 3};

CREATE TABLE chat_system.messages (
    channel_id UUID,
    bucket_month INT, -- Format: YYYYMM (e.g., 202609)
    message_id TIMEUUID, -- Guarantees monotonic time ordering
    sender_id UUID,
    encrypted_content BLOB,
    PRIMARY KEY ((channel_id, bucket_month), message_id)
) WITH CLUSTERING ORDER BY (message_id ASC);

-- PostgreSQL Channel & User Session Metadata
CREATE TABLE channel_members (
    channel_id UUID NOT NULL,
    user_id UUID NOT NULL,
    role VARCHAR(16) NOT NULL DEFAULT 'MEMBER',
    last_read_message_id UUID,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (channel_id, user_id)
);

CREATE TABLE user_sessions (
    user_id UUID NOT NULL,
    device_id UUID NOT NULL,
    gateway_node_ip INET NOT NULL,
    last_heartbeat TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id, device_id)
);

CREATE INDEX idx_channel_users ON channel_members(user_id);
```

#### Step-by-Step Execution Sequence
1. **Connection & Presence Registration:** Client establishes persistent WSS connection to Gateway pod. Gateway registers `(user_id, gateway_node_ip)` in Redis and sets user bit in Redis presence bitmap.
2. **Message Send & Monotonic Sequencing:** Client transmits encrypted message. Gateway assigns monotonic `TIMEUUID` and appends payload to Cassandra partition `(channel_id, bucket_month)`.
3. **Recipient Routing Lookup:** Gateway checks Redis session map for channel members:
   - For **online members**: routes payload directly to their designated gateway pod via gRPC for immediate WebSocket push.
   - For **offline members**: enqueues notification job into Kafka topic `push.notifications`.
4. **Push Notification Dispatch:** Mobile push worker pulls message, constructs APNs/FCM payload, and delivers alert to recipient device.
5. **Delivery & Read Acknowledgments:** Recipient sends ACK over WebSocket. Gateway updates `last_read_message_id` in PostgreSQL and broadcasts read receipt to channel.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **WebSocket Gateway Node Crash** | Server dies; 150,000 mobile sockets disconnect, triggering synchronized reconnect stampede. | **Randomized Reconnect Jitter with Backoff:** Mobile SDKs implement exponential backoff with full jitter ($1\text{--}30\text{s}$). Gateway pods fronted by AWS Network Load Balancer (NLB) with cross-zone load balancing. |
| **Mega-Group Broadcast Storm** | User posts in 100,000-member community channel; naive iteration spawns 100,000 immediate gRPC calls. | **Publish-Subscribe Fan-Out with Read-on-Open:** Messages are published to a single Redis Pub/Sub channel. Only members currently viewing the screen receive live deltas; all others fetch on next view. |
| **Network Switch Causes Out-of-Order Delivery**| Mobile device switches from 5G to Wi-Fi mid-stream, delivering older message after newer message. | **Client Local Sequence Buffer:** Mobile app buffers incoming messages in local SQLite by `TIMEUUID`. UI renders messages strictly sorted by `TIMEUUID` regardless of TCP packet arrival order. |
| **Cassandra Hot Partition Blowup** | Active channel with millions of messages exceeds 100MB Cassandra partition threshold. | **Composite Monthly Partition Bucketing:** Primary key combines `(channel_id, bucket_month)`. A new partition is automatically initialized each month, strictly bounding partition sizes below 50 MB. |

#### Staff-Level Interview Verbalization
> *"Our messaging architecture handles 1.5 million messages per second by decoupling stateful edge connections from distributed storage. We terminate persistent TLS WebSockets on epoll-driven Netty gateways and persist messages into ScyllaDB using a composite partition key of channel ID and monthly bucket, ordered monotonically by TIMEUUID to eliminate partition bloat. User presence is tracked across 500 million users in just 60 megabytes of RAM using Redis Bitmaps, while mega-group fan-out storms are mitigated via hybrid Pub/Sub and read-on-open delivery."*


### Solution 10: Task Scheduler — Distributed Workflow & Job Scheduler Engine (Temporal / Airflow)

#### Problem Statement & SLAs
Design a distributed task scheduler and workflow orchestration engine capable of executing delayed, recurring, and dependent DAG jobs.

- **Scale:** 100 million scheduled tasks/day ($\approx 10,000 \text{ executions/sec}$ peak).
- **Execution SLA:** Task execution delay $< 500\text{ms}$ from scheduled target time ($p99 < 100\text{ms}$).
- **Reliability SLA:** At-least-once execution guarantee with idempotency tokens. Zero duplicate executions.

#### Capacity Estimation & Hardware Sizing
- **Task Payload:** 2 KB task context payload.
- **Daily State Storage:** $100 \times 10^6 \text{ tasks/day} \times 2 \text{ KB} = 200 \text{ GB/day}$ state log; $6 \text{ TB/month}$.
- **Throughput:** Sustained 1,160 tasks/sec; peak burst 10,000 tasks/sec.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Scheduler Master Nodes** | 3 $\times$ `c6i.2xlarge` (1 Leader + 2 Standby) | 8 vCPU, 16 GB RAM | etcd consensus for leader election, active DAG dependency evaluation |
| **Hierarchical Timing Wheels** | 16 $\times$ `r6i.2xlarge` (EKS Workers) | 8 vCPU, 64 GB RAM per node | In-memory 5-level timing wheels for $\mathcal{O}(1)$ delayed task triggers |
| **Distributed Worker Pool** | 60 $\times$ `c6i.4xlarge` (Auto-Scaling HPA)| 16 vCPU, 32 GB RAM | Pulls ready tasks from priority Kafka/RabbitMQ topics, runs task logic |
| **Task State Database** | 1 Primary + 2 Read Replicas | `db.r6i.4xlarge` (128 GB RAM) | Aurora PostgreSQL, daily range partitioned task history and dependency graphs |
| **Lock & Lease Manager** | 5-node `etcd` / ZooKeeper Cluster | 4 vCPU, 16 GB RAM, NVMe | Distributed leasing, fencing tokens, and dynamic worker heartbeat leases |

#### Visual Architecture Blueprint
![Figure 17.10: Distributed Task Scheduler & Workflow Engine Architecture](visuals/arch_task_scheduler_workflow.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Hierarchical Timing Wheel (Delayed Scheduling Engine):**
   - Min-heap priority queues require $\mathcal{O}(\log N)$ insertion and deletion. For 100 million tasks, $\log_2(10^8) \approx 27$ pointer operations per insert, causing severe CPU cache thrashing.
   - **Hierarchical Timing Wheels (Varghese & Lauck)** use multi-tier circular ring buffers (millisecond, second, minute, hour, day wheels) with fixed slot arrays:
     - As the pointer advances in $\mathcal{O}(1)$ time, tasks scheduled for the current tick fire immediately.
     - When the hour wheel advances by one slot, remaining tasks cascade down into the minute wheel, then to the second wheel, guaranteeing $\mathcal{O}(1)$ amortized insertion and trigger.
2. **Distributed Fencing Tokens (Split-Brain Prevention):**
   - To eliminate zombie worker hazards (where a worker pauses due to a 20-second JVM GC pause, loses its lock lease, but wakes up and blindly commits side-effects), the lock manager issues a monotonic **Fencing Token** with every lease.
   - Any database update or external service call must supply `WHERE fencing_token >= current_token`. If a new worker has been granted token 43, worker 42's write is rejected with an optimistic lock violation.
3. **DAG Dependency Resolution:**
   - Evaluates Directed Acyclic Graphs (DAGs) using topological sorting and in-degree counters.
   - When a parent task completes, the orchestrator decrements the `unmet_dependencies` counter of child tasks; when counter reaches 0, the child task transitions to `READY`.

#### API Contracts & Interface Specs (gRPC Protobuf)
```protobuf
syntax = "proto3";
package scheduler;

message SubmitWorkflowRequest {
    string idempotency_key = 1;
    string workflow_name = 2;
    repeated TaskDefinition tasks = 3;
    map<string, string> input_params = 4;
}

message TaskDefinition {
    string task_id = 1;
    string task_type = 2;
    int64 delay_seconds = 3;
    repeated string depends_on = 4; // Parent task IDs
    int32 max_retries = 5;
    int64 timeout_seconds = 6;
}

message TaskExecutionClaim {
    string task_id = 1;
    string worker_id = 2;
    int64 lease_duration_ms = 3;
}

message TaskExecutionClaimResponse {
    bool acquired = 1;
    int64 fencing_token = 2;
    string payload_json = 3;
}
```

#### Production Database Schema & Data Model (PostgreSQL DDL)
```sql
-- Partitioned task execution history
CREATE TABLE tasks (
    task_id UUID NOT NULL,
    workflow_id UUID NOT NULL,
    task_type VARCHAR(64) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('PENDING','READY','RUNNING','COMPLETED','FAILED','RETRY')),
    fencing_token BIGINT NOT NULL DEFAULT 0,
    scheduled_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    worker_id VARCHAR(128),
    retry_count INT NOT NULL DEFAULT 0,
    max_retries INT NOT NULL DEFAULT 3,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, scheduled_at)
) PARTITION BY RANGE (scheduled_at);

-- Daily partition
CREATE TABLE tasks_2026_09_01 PARTITION OF tasks
    FOR VALUES FROM ('2026-09-01 00:00:00+00') TO ('2026-09-02 00:00:00+00');

CREATE INDEX idx_tasks_pending ON tasks(scheduled_at) 
    WHERE status IN ('PENDING', 'READY');

CREATE TABLE task_dependencies (
    parent_task_id UUID NOT NULL,
    child_task_id UUID NOT NULL,
    workflow_id UUID NOT NULL,
    PRIMARY KEY (parent_task_id, child_task_id)
);
```

#### Step-by-Step Execution Sequence
1. **Workflow DAG Ingest:** User submits workflow via gRPC. Orchestrator validates graph for cycles, persists task records in PostgreSQL, and schedules initial root tasks.
2. **Timing Wheel Enqueue:** Delayed tasks are inserted into the memory ring buffer based on execution timestamp ($\mathcal{O}(1)$ slot insert).
3. **Trigger & Claim with Fencing Token:** When timer advances, task fires. Worker attempts atomic claim: `UPDATE tasks SET status = 'RUNNING', worker_id = ?, fencing_token = fencing_token + 1 WHERE task_id = ? AND status = 'READY' RETURNING fencing_token`.
4. **Execution & Heartbeating:** Worker executes job, extending lease heartbeat every 10 seconds.
5. **Child Task Cascade or DLQ:** On success, orchestrator marks task `COMPLETED` and decrements dependent tasks' in-degree counters. On failure, applies exponential backoff with full jitter; if retries exceed 3, routes payload to Dead Letter Queue (DLQ).

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Zombie Worker / Split-Brain Commit** | Worker pauses during 20s GC pause; lease expires and reassigns task. Paused worker wakes up and executes duplicate transaction. | **Monotonic Fencing Tokens:** Database enforces `WHERE fencing_token = :claimed_token`. The reassigned worker has incremented the token; the zombie worker's update is immediately rejected. |
| **Timing Wheel Memory Exhaustion** | 50 million tasks scheduled for 6 months in the future fill ring buffer RAM. | **Cold Tier Staging:** Timing wheels only load tasks scheduled for the next 24 hours. A background poller streams upcoming tasks from PostgreSQL into the wheel on a 1-hour rolling window. |
| **Poison Pill Task Cascades** | Corrupted task crashes worker repeatedly, triggering infinite retry storm across cluster. | **Exponential Backoff with Jitter & DLQ:** Retries calculate delay as $T = \min(M, 2^{\text{retry}} \cdot B) \pm \text{jitter}$. After 3 failures, task is quarantined into DLQ without killing worker pods. |
| **Scheduler Master Node Failure** | Active leader scheduler pod crashes mid-evaluation. | **etcd Distributed Lease Election:** Standby scheduler nodes observe missing heartbeat within 1 second, elect new leader via etcd Raft, and resume active queue evaluation from PostgreSQL state. |

#### Staff-Level Interview Verbalization
> *"Our task scheduler handles 100 million daily jobs with sub-100ms precision by replacing conventional min-heap priority queues with Hierarchical Timing Wheels, achieving $\mathcal{O}(1)$ insertion and expiration complexity. We eliminate the classic distributed zombie worker problem by issuing monotonic fencing tokens with every lock lease, guaranteeing that delayed or partitioned workers can never commit duplicate operations. Workflows evaluate as dependency-counted DAGs, and long-term scheduled tasks stage through a tiered cold-store to protect in-memory ring buffers."*


### Solution 11: Collaborative Editor — Real-Time CRDT & Whiteboard Engine (Figma / Google Docs / Notion)

#### Problem Statement & SLAs
Design a real-time collaborative document editor and interactive whiteboard allowing concurrent editing by hundreds of users per document.

- **Scale:** 50,000 active concurrent editing sessions (up to 500 concurrent collaborators per canvas/document).
- **Latency SLA:** Local keypress edit rendering $0\text{ms}$ (instant optimistic UI). Remote peer sync $p99 < 50\text{ms}$.
- **Consistency SLA:** Strong Eventual Consistency (SEC) — all connected clients converge to identical document states without a central locking authority.

#### Capacity Estimation & Hardware Sizing
- **Real-Time Cursor Ingest:** $50,000 \text{ active sessions} \times 10 \text{ updates/sec} = 500,000 \text{ messages/sec}$.
- **Bandwidth:** $500,000 \text{ msgs/sec} \times 64 \text{ bytes} = 32 \text{ MB/sec} = 256 \text{ Mbps}$.
- **Document Edit Operations:** 25,000 write ops/sec peak.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **WebSocket Gateway (WSS)** | 20 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Maintains persistent client session WebSockets, streams delta operations |
| **CRDT State Sync Workers** | 24 $\times$ `c6i.2xlarge` (EKS Pods) | 8 vCPU, 16 GB RAM | Evaluates binary Yjs/Automerge state vector diffs, coordinates snapshots |
| **Cursor Awareness Pub/Sub**| 8 $\times$ `r6i.xlarge` (Redis Cluster) | 4 vCPU, 32 GB RAM per node | Ephemeral Pub/Sub channels for live cursor positions at 30Hz; zero disk I/O |
| **Document State DB** | 1 Primary + 1 Replica | `db.r6i.2xlarge` (64 GB RAM) | Aurora PostgreSQL for document ACLs, workspace metadata, and snapshot pointers |
| **Snapshot Block Store** | AWS S3 Standard | Object Storage | Houses immutable zstd-compressed full document snapshots every 1,000 ops |

#### Visual Architecture Blueprint
![Figure 17.11: Real-Time Collaborative Document Editor Architecture](visuals/arch_collaborative_crdt_editor.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Conflict Resolution Strategy (CRDT vs. OT):**
   - **Operational Transformation (OT):** Requires a centralized server to serialize all operations and transform concurrent operations against each other ($op_1 \circ op_2'$). Network partitions or server disconnections stall the entire editing loop.
   - **Operation-Based CRDT (Yjs / Automerge / RGA):** Every character or vector shape is assigned an immutable globally unique identifier:
     $$\text{ID} = (\text{client\_id}, \text{lamport\_clock}, \text{fractional\_index})$$

   - Operations form a bounded join-semilattice where mutations are commutative, associative, and idempotent ($A \sqcup B = B \sqcup A$). Peers converge to the exact same document tree deterministically without locks or central arbitration.
2. **Fractional Indexing for Dense Positioning:**
   - Inserting a character between two adjacent items (e.g., between position `1.0` and `2.0`) does not re-index the array. Instead, it generates an infinite-precision rational midpoint (e.g., `1.5`, or `1.25` on further insert), guaranteeing $\mathcal{O}(1)$ insertion complexity.
3. **State Vector Synchronization:**
   - When a client connects, it transmits a compact **State Vector** summarizing its observed clock per client:
     $$SV = \{\text{client}_A: 104, \text{client}_B: 88\}$$

   - The sync server compares vectors and transmits only missing binary deltas, reducing initial sync payloads by $> 95\%$.
4. **Ephemeral Awareness vs. Persistent State Separation:**
   - Mouse cursor coordinates, live text selections, and user avatars broadcast via ephemeral Redis Pub/Sub channels at 30Hz, completely bypassing database persistence.
   - Core text and vector shape CRDT operations persist into the PostgreSQL operation log.

#### API Contracts & Interface Specs
```json
// WebSocket: wss://collab.example.com/doc/{doc_id}
// Client -> Server (CRDT Operation Delta):
{
  "type": "SYNC_STEP_1",
  "state_vector": "0102a9128f...",
  "doc_id": "doc_99182a"
}
// Server -> Client (Missing Operation Delta):
{
  "type": "SYNC_STEP_2",
  "update_binary_base64": "AQAxODJh...",
  "server_seq": 10482
}
// Client -> Server (Ephemeral Cursor Broadcast):
{
  "type": "AWARENESS_UPDATE",
  "user_id": "usr_42",
  "cursor": {"x": 482.5, "y": 120.0},
  "color": "#FF5733"
}
```

#### Production Database Schema & Data Model (PostgreSQL + S3 DDL)
```sql
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL,
    owner_id UUID NOT NULL,
    title VARCHAR(512) NOT NULL,
    latest_server_seq BIGINT NOT NULL DEFAULT 0,
    current_snapshot_s3_key VARCHAR(512),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_operations (
    doc_id UUID NOT NULL,
    server_seq BIGINT NOT NULL,
    client_id VARCHAR(64) NOT NULL,
    lamport_clock BIGINT NOT NULL,
    op_type VARCHAR(16) NOT NULL CHECK (op_type IN ('INSERT','DELETE','FORMAT','MOVE')),
    op_binary BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, server_seq)
);

CREATE TABLE document_snapshots (
    snapshot_id UUID PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id),
    server_seq BIGINT NOT NULL,
    s3_key VARCHAR(512) NOT NULL,
    byte_size INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_doc_ops ON document_operations(doc_id, server_seq DESC);
```

#### Step-by-Step Execution Sequence
1. **Optimistic Local Mutation:** User inputs character or drags canvas element. Client updates local DOM/WebGL canvas instantly ($0\text{ms}$ latency) and creates local CRDT operation node.
2. **WebSocket Delta Transmission:** Client serializes delta into binary format (Lib0 / Protobuf) and transmits over WebSocket to Gateway.
3. **Sequence Allocation & Redis Pub/Sub Broadcast:** Gateway stamps monotonic `server_seq`, appends to PostgreSQL operation log, and broadcasts delta to Redis Pub/Sub channel `doc:{id}`.
4. **Peer CRDT Merge:** Peer clients receive binary delta, unpack operations, and merge them into their local CRDT document model. Commutative properties guarantee identical document convergence.
5. **Periodic Snapshot Compaction:** When `server_seq % 1000 == 0`, worker generates a zstd-compressed document snapshot, uploads to S3, and updates `current_snapshot_s3_key`.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Operation Log Explosion** | Document with 1 million edits requires 10 minutes to load and replay on new client join. | **Periodic S3 Snapshot Compaction:** Worker collapses operation log every 1,000 edits into an immutable S3 snapshot. New clients download base snapshot and replay only subsequent deltas ($< 1,000$ operations). |
| **Awareness Broadcast Storm** | 500 users in a shared canvas move cursors simultaneously, generating 15,000 updates/sec and choking client WebSockets. | **Client Throttling & Spatial Proximity Filter:** Client SDK throttles cursor broadcasts to 30Hz. Gateway filters awareness updates, transmitting cursor movements only to collaborators viewing the same screen viewport. |
| **Prolonged Offline Split-Brain** | Designer edits canvas on flight for 6 hours; reconnects with 20,000 uncommitted edits. | **Binary State Vector Diff Merge:** On reconnect, client exchanges state vector with server. CRDT mathematical semilattice resolves all edits deterministically without overwriting concurrent work. |
| **In-Memory Document Leak** | Abandoned document sessions remain cached in Gateway RAM, leading to node OOM. | **Inactivity Eviction Lease:** Document memory state is assigned a 10-minute sliding lease. If all WebSockets disconnect, the state flushes to S3/Postgres and cleans up RAM immediately. |

#### Staff-Level Interview Verbalization
> *"Our collaborative editor platform guarantees Strong Eventual Consistency with zero UI latency using Operation-Based CRDTs. Mutations execute optimistically on the client DOM in 0 milliseconds and fan out over WebSockets as compact binary deltas tagged with Lamport timestamps and fractional indices. By separating high-frequency ephemeral mouse movements into Redis Pub/Sub from persistent document state in PostgreSQL and S3 snapshots, we support hundreds of concurrent collaborators per document without central lock contention."*


### Solution 12: Observability — Distributed Time-Series Metrics Platform (Prometheus / Datadog / Grafana)

#### Problem Statement & SLAs
Design a distributed time-series database (TSDB) and observability platform for ingesting system metrics, generating alerts, and serving dashboards.

- **Scale:** 10 million active time-series metrics ingested every 10 seconds ($\approx 1 \text{ million metric data points/sec}$).
- **Query SLA:** PromQL dashboard query execution $p99 < 200\text{ms}$ over 2-hour hot ranges.
- **Retention & Durability:** Raw metrics stored for 14 days; downsampled 5-minute rollups stored for 1 year. Zero lost metric blocks.

#### Capacity Estimation & Hardware Sizing
- **Uncompressed Metric Point:** 16 bytes (8B timestamp + 8B IEEE 754 float value).
- **Gorilla Compression Ratio:** Compresses 16 bytes down to an average of **1.37 bytes** per sample ($11.6 \times$ memory reduction).
- **Daily Storage Ingress:** $1 \times 10^6 \text{ samples/sec} \times 86,400 \text{ sec/day} \times 1.37 \text{ bytes} \approx 118.3 \text{ GB/day}$.
- **In-Memory Head Buffer RAM:** 2 hours of hot samples $\approx 10^6 \times 7,200 \times 1.37 \text{ B} \approx 9.86 \text{ GB RAM}$ net; $\approx 45 \text{ GB RAM}$ with series label index overhead.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Ingestion Gateways** | 30 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Terminates 1M samples/sec push/pull, validates labels, routes to shards |
| **TSDB Storage Nodes (Head)** | 20 $\times$ `r6i.4xlarge` (StatefulSet) | 16 vCPU, 128 GB RAM, NVMe | Houses 2-hour in-memory Gorilla head chunks, flushes immutable 2h WAL blocks |
| **Downsampling Aggregators** | 16 $\times$ `c6i.2xlarge` (EKS Workers) | 8 vCPU, 16 GB RAM | Computes 1-minute and 5-minute rollups (min, max, sum, count) asynchronously |
| **PromQL Query Routers** | 12 $\times$ `c6i.4xlarge` (EKS) | 16 vCPU, 32 GB RAM | Evaluates PromQL AST, parallelizes chunk scans across head nodes and S3 |
| **Long-Term Block Store** | AWS S3 Standard + Glacier | Object Storage | Houses compressed 2-hour historical TSDB blocks and downsampled parquet |

#### Visual Architecture Blueprint
![Figure 17.12: Distributed Time-Series Metrics & Observability Platform Architecture](visuals/arch_metrics_timeseries_observability.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Metrics Collection & Ingress (Push / Pull):**
   - Application pods and infrastructure exporters (`node_exporter`, OpenTelemetry Collector) stream metric samples over HTTP/gRPC.
   - Consistent hashing on `(metric_name, label_hash)` routes series deterministically to TSDB storage nodes.
2. **Gorilla TSDB Compression (Facebook VLDB 2015):**
   - **Timestamps (Delta-of-Delta):**
     $$D = (t_i - t_{i-1}) - (t_{i-1} - t_{i-2})$$

     - If $D = 0$: Store single bit `0` (accounts for $> 96\%$ of regular interval metric points).
     - If $-63 \le D \le 64$: Store bits `10` followed by 7 bits of value.
     - If $-255 \le D \le 256$: Store bits `110` followed by 9 bits.
     - If $-2047 \le D \le 2048$: Store bits `1110` followed by 12 bits.
     - Otherwise: Store bits `1111` followed by full 32-bit delta.
   - **Floating-Point Values (XOR Encoding):**
     $$X = V_i \oplus V_{i-1}$$

     - If $X = 0$ (identical value): Store single bit `0`.
     - If $X \ne 0$: Store bit `1`. If the leading and trailing zero counts match the previous sample, store `0` + meaningful bits. Otherwise, store `1` + (5 bits leading count) + (6 bits length) + meaningful bits.
     - Compresses 64-bit IEEE 754 floats to an average of **1.37 bytes/sample**.
3. **Inverted Label Index & Roaring Bitmaps:**
   - Labels are indexed using an inverted posting list: `tag_name:tag_value -> RoaringBitmap(series_ids)`.
   - Querying `{app="payment", status="500"}` performs blazing-fast bitwise AND (`&`) across Roaring Bitmaps, resolving matching series IDs in under $2\text{ms}$.
4. **Tiered Storage & Rollup Downsampling:**
   - 2-hour in-memory head chunks flush to NVMe disk as immutable TSDB blocks.
   - Downsampling workers collapse raw 10-second data into 5-minute min/max/sum/count rollups for long-term historical trend queries, reducing 1-year storage requirements by $96\%$.

#### API Contracts & Interface Specs
```json
// POST /api/v1/query_range (PromQL Query)
Header: Content-Type: "application/json"
{
  "query": "sum(rate(http_requests_total{status=~'5..'}[5m])) by (service)",
  "start": "2026-08-30T10:00:00Z",
  "end": "2026-08-30T12:00:00Z",
  "step": "15s"
}
// Response (200 OK):
{
  "status": "success",
  "data": {
    "resultType": "matrix",
    "result": [
      {
        "metric": {"service": "payment-api"},
        "values": [
          [1725012000, "14.2"],
          [1725012015, "12.8"],
          [1725012030, "15.1"]
        ]
      }
    ]
  }
}
```

#### Production Database Schema & Data Model (TSDB + PostgreSQL DDL)
```sql
-- Series metadata registry
CREATE TABLE metric_series (
    series_id BIGINT PRIMARY KEY,
    metric_name VARCHAR(128) NOT NULL,
    labels JSONB NOT NULL,
    fingerprint BIGINT NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL,
    last_seen TIMESTAMPTZ NOT NULL
);

-- Fast label matching via PostgreSQL GIN index
CREATE INDEX idx_series_labels ON metric_series USING GIN (labels);
CREATE INDEX idx_series_metric ON metric_series (metric_name);

-- Rollup aggregated table for downsampled historical queries
CREATE TABLE metric_rollups_5m (
    series_id BIGINT NOT NULL,
    bucket_time TIMESTAMPTZ NOT NULL,
    sample_count INT NOT NULL,
    val_min DOUBLE PRECISION NOT NULL,
    val_max DOUBLE PRECISION NOT NULL,
    val_sum DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (series_id, bucket_time)
) PARTITION BY RANGE (bucket_time);

-- Monthly partition for 5m rollups
CREATE TABLE metric_rollups_2026_09 PARTITION OF metric_rollups_5m
    FOR VALUES FROM ('2026-09-01 00:00:00+00') TO ('2026-10-01 00:00:00+00');
```

#### Step-by-Step Execution Sequence
1. **Sample Push & Shard Routing:** Exporter sends metric payload. Gateway computes consistent hash of metric labels and routes sample to designated TSDB node.
2. **Gorilla Append in Head Chunk:** TSDB node looks up series ID. Computes Delta-of-Delta timestamp and XOR float bit encoding, appending bits into an active 2-hour in-memory chunk.
3. **WAL Logging:** Appends uncompressed batch to sequential Write-Ahead Log on NVMe SSD to survive power loss.
4. **2-Hour Chunk Cut & Seal:** When 2 hours elapse, node seals the in-memory head chunk, creates immutable disk block with Lucene-style index, and resets head buffer.
5. **Downsampling & S3 Flush:** Background worker reads completed blocks after 24 hours, downsamples samples into 5-minute summaries, and flushes immutable blocks to S3.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **High-Cardinality Metric Explosion** | Developer mistakenly includes `user_id` in metric labels, creating 50M new series and crashing label RAM. | **Dynamic Ingestion Rate Limiting & Series Dropping:** Ingestion gateway monitors unique series creation rate per tenant. If series count exceeds quota ($100\text{k}/\text{tenant}$), new ephemeral labels are stripped and alert fires. |
| **TSDB Head Chunk Memory Exhaustion** | Ingestion burst causes in-memory Gorilla buffer to exceed 85% node RAM. | **Early Head Truncation & Proactive Disk Flush:** Storage node monitors memory watermarks. If RAM $> 85\%$, node forces an early 2-hour chunk cut, seals the buffer, and flushes to NVMe SSD immediately. |
| **Heavy PromQL Query Exhaustion** | User executes `count(http_requests) by (ip)` over a 6-month interval, threatening query engine OOM. | **Query Planner Sample Limits & Rollup Routing:** Query router enforces maximum $100,000$ data points scanned limit. Queries spanning $> 7\text{ days}$ are transparently rewritten to query 5-minute precomputed rollups. |
| **Collector Network Partition** | Network disconnects Kubernetes cluster from central metrics store for 15 minutes. | **Local Agent Buffer & Spooling:** OpenTelemetry / Prometheus agent pods maintain an in-memory and local disk ring buffer, buffering up to 15 minutes of samples with exponential reconnect backoff. |

#### Staff-Level Interview Verbalization
> *"Our observability architecture scales to 1 million metrics per second by combining Facebook Gorilla compression with an inverted label index backed by Roaring Bitmaps. Timestamps compress via Delta-of-Delta encoding and floats via bitwise XOR, reducing storage to an average of 1.37 bytes per sample. We separate hot queries (served from 2-hour in-memory head chunks) from long-term analytical queries (served from asynchronously downsampled 5-minute rollups in S3), protecting the cluster against high-cardinality label explosions via dynamic ingestion quotas."*


### Solution 13: Notifications — Distributed Multi-Channel Notification & Alerting Platform

#### Problem Statement & SLAs
Design a multi-channel notification platform supporting Email, SMS, Push (APNs/FCM), and In-App WebSocket delivery with deduplication and user preference management.

- **Scale:** 1 billion notifications/day ($\approx 12,000 \text{ notifications/sec}$ sustained, $50,000/\text{sec}$ peak).
- **Delivery SLA:** Push/In-App delivery $p99 < 500\text{ms}$. SMS delivery $p99 < 5\text{s}$. Email delivery $p99 < 30\text{s}$.
- **Deduplication SLA:** Zero duplicate notifications to the same recipient for the same event within a 24-hour window.

#### Capacity Estimation & Hardware Sizing
- **Notification Payload:** Average 1 KB per notification (template ID + user context + channel metadata).
- **Daily Storage:** $1 \times 10^9 \text{ notifications/day} \times 1 \text{ KB} = 1 \text{ TB/day}$ delivery log; $30 \text{ TB/month}$.
- **Redis Bloom Filter (Dedup Math):**
  - Optimal bit array size $m$:
    $$m = -\frac{n \ln p}{(\ln 2)^2} = -\frac{10^9 \cdot \ln(0.001)}{(0.6931)^2} \approx 14.37 \text{ billion bits} \approx 1.79 \text{ GB RAM}$$

  - Optimal number of hash functions $k$:
    $$k = \frac{m}{n} \ln 2 = \frac{14.37 \times 10^9}{10^9} \times 0.6931 \approx 10 \text{ hash functions}$$

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Notification API Gateway** | 20 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Validates payload, checks Bloom filter, routes to priority Kafka queues |
| **Bloom Deduplication Cache**| 6 $\times$ `r6i.xlarge` (Redis Cluster) | 4 vCPU, 32 GB RAM per node | Houses daily Redis Bloom filters for 1B notifications in ~1.79 GB RAM |
| **Priority Kafka Cluster** | 8 $\times$ `i3en.2xlarge` (KRaft) | 8 vCPU, 64 GB RAM, NVMe | Independent topics for `HIGH` (OTP), `MEDIUM` (trans), `LOW` (promo) |
| **Template & Rendering Pods** | 16 $\times$ `c6i.xlarge` (EKS Workers) | 4 vCPU, 8 GB RAM | Hydrates Mustache/Jinja templates with recipient preferences and i18n locales |
| **Downstream Adapters** | 40 $\times$ `c6i.xlarge` (EKS Workers) | 4 vCPU, 8 GB RAM | Dedicated adapter pools: SES (12), Twilio (10), APNs/FCM (18) |
| **Notification Audit Log DB** | 1 Primary + 3 Read Replicas | `db.r6i.4xlarge` (128 GB RAM) | Aurora PostgreSQL, monthly partitioned delivery logs with GIN indexes |

#### Visual Architecture Blueprint
![Figure 17.13: Distributed Multi-Channel Notification & Alerting Platform](visuals/arch_notification_platform.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Ingress & Edge Deduplication:**
   - API Gateway intercepts requests and evaluates idempotency keys against a **Redis Bloom Filter** ($m=14.37\text{B bits}, k=10$).
   - If the bit is set, the gateway queries PostgreSQL to confirm whether the notification was already processed. If duplicate, returns `200 OK` without resending.
2. **Priority Queue Isolation (Starvation Prevention):**
   - Critical transactional messages (e.g., 2-Factor Authentication OTP SMS) must never queue behind a 10-million recipient marketing newsletter blast.
   - Kafka topics are split into strict priority tiers:
     - `notifications.high`: Dedicated partitions for OTPs and fraud alerts (consumed immediately).
     - `notifications.medium`: Account updates, order confirmations, shipment tracking.
     - `notifications.low`: Promotional blasts, weekly digests, recommendations.
3. **Template Rendering & User Preference Engine:**
   - Evaluates recipient preferences: verifies enabled channels, quiet hours (e.g., suppress promotional SMS between 9 PM and 8 AM in recipient timezone), and unsubscriptions.
   - Hydrates templates using cached Mustache models.
4. **Third-Party Provider Adapters & Circuit Breaking:**
   - Channel adapters communicate with downstream providers (AWS SES for Email, Twilio / Sinch for SMS, APNs / FCM for Push, WebSockets for In-App).
   - Each provider is protected by a circuit breaker. If Twilio returns 503 or throttles, the SMS adapter trips and routes traffic to a secondary provider (e.g., MessageBird / Sinch).

#### API Contracts & Interface Specs
```json
// POST /v1/notifications/send (Single Notification)
Header: X-Idempotency-Key: "evt_payment_confirmed_usr42_20260901"
{
  "recipient_id": "usr_88102",
  "template_id": "tmpl_payment_success",
  "priority": "HIGH",
  "channels": ["PUSH", "EMAIL"],
  "context": {
    "amount": "$129.99",
    "order_id": "ord_7712",
    "customer_name": "Sarah"
  }
}
// Response (202 Accepted):
{
  "notification_id": "ntf_a91f2",
  "status": "QUEUED",
  "estimated_delivery_ms": 320
}
```

#### Production Database Schema & Data Model (PostgreSQL DDL)
```sql
CREATE TABLE notification_templates (
    template_id VARCHAR(64) PRIMARY KEY,
    channel VARCHAR(16) NOT NULL,
    subject_template TEXT,
    body_template TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY,
    email_enabled BOOLEAN NOT NULL DEFAULT true,
    sms_enabled BOOLEAN NOT NULL DEFAULT true,
    push_enabled BOOLEAN NOT NULL DEFAULT true,
    timezone VARCHAR(64) NOT NULL DEFAULT 'UTC',
    quiet_hours_start TIME,
    quiet_hours_end TIME
);

-- Partitioned audit log table
CREATE TABLE notification_log (
    notification_id UUID NOT NULL,
    user_id UUID NOT NULL,
    idempotency_key VARCHAR(128) NOT NULL,
    template_id VARCHAR(64) NOT NULL,
    channel VARCHAR(16) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('QUEUED','SENT','DELIVERED','FAILED','BOUNCED','SUPPRESSED')),
    priority VARCHAR(10) NOT NULL DEFAULT 'MEDIUM',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMPTZ,
    PRIMARY KEY (notification_id, created_at)
) PARTITION BY RANGE (created_at);

-- Monthly partition
CREATE TABLE notification_log_2026_09 PARTITION OF notification_log
    FOR VALUES FROM ('2026-09-01 00:00:00+00') TO ('2026-10-01 00:00:00+00');

CREATE INDEX idx_notif_user ON notification_log(user_id, created_at DESC);
CREATE INDEX idx_notif_idemp ON notification_log(idempotency_key);
```

#### Step-by-Step Execution Sequence
1. **Idempotency Verification:** Gateway receives request, evaluates Redis Bloom Filter. If new, adds key to filter and forwards to Priority Queue Router.
2. **Priority Topic Dispatch:** Router assigns message to `notifications.high` or `notifications.low` Kafka topic based on priority payload header.
3. **Preference & Quiet Hours Filter:** Worker queries `user_preferences`. If current time falls within user quiet hours and priority is not `HIGH`, schedules task for delivery at end of quiet window.
4. **Template Hydration:** Worker loads body template from Redis cache, compiles variables, and constructs localized payload.
5. **Channel Adapter Dispatch & Fallback:** Worker calls provider API (e.g., Twilio). On HTTP 200, updates status to `DELIVERED`. On timeout or error, trips circuit breaker to secondary provider (Sinch) and logs state transition in PostgreSQL.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Primary SMS Provider Outage** | Twilio suffers multi-region outage; 2FA login codes fail globally, locking out millions of users. | **Automated Multi-Provider Failover:** SMS adapter uses Resilience4j circuit breaker. If error rate $> 15\%$, traffic automatically routes to secondary provider (Sinch) within 2 seconds. |
| **Priority Inversion / OTP Starvation** | Marketing blast of 50M Black Friday promotional emails delays password reset tokens by 45 minutes. | **Strict Kafka Queue Isolation:** OTPs and security alerts run on independent Kafka topics with dedicated, unshared consumer worker pools. Marketing blasts are rate-limited to 5,000 msg/sec. |
| **Concurrent Double Send** | Mobile app retries payment confirmation alert after 2-second timeout; user receives two identical emails. | **Redis Bloom Filter + DB Idempotency Key:** Redis Bloom filter provides instant early rejection. The `idempotency_key` unique constraint in PostgreSQL guarantees exactly-once dispatch at the database level. |
| **User Opt-Out In-Flight Race** | User unsubscribes while a 500,000-email batch is actively processing in worker memory. | **Just-in-Time Preference Check:** Channel adapter performs a lightweight cache check against `user:suppression_list` immediately before dispatching the payload to the external provider. |

#### Staff-Level Interview Verbalization
> *"Our notification platform handles 1 billion messages daily by decoupling ingestion from channel execution via priority-partitioned Kafka topics. This prevents high-volume marketing blasts from starving mission-critical 2FA OTP codes. We eliminate duplicate notifications across 1 billion items in less than 2 gigabytes of RAM using Redis Bloom Filters paired with database idempotency constraints. Each channel adapter is fortified with circuit breakers that automatically fail over to secondary providers during downstream outages, while respecting recipient timezones and quiet hour preferences."*


### Solution 14: Booking Engine — Distributed Hotel & Flight Inventory Reservation System (Airbnb / Booking.com)

#### Problem Statement & SLAs
Design a distributed inventory reservation system for hotels and flights that prevents double-booking under concurrent access from millions of users.

- **Scale:** 50,000 concurrent booking sessions, 5,000 reservations/minute peak.
- **Consistency SLA:** Strict ACID consistency — an inventory room-night or flight seat sold to one customer is never simultaneously sold to another. Zero double bookings.
- **Latency SLA:** Search availability $p99 < 100\text{ms}$. End-to-end reservation confirmation $p99 < 2\text{s}$ (including external payment).

#### Capacity Estimation & Hardware Sizing
- **Inventory Units:** 10 million hotel rooms + 500,000 flights $\times$ 365 days = $\approx 3.8 \text{ billion calendar-day slots}$.
- **Calendar Slot Size:** 64 bytes per slot (room\_id, date, status, reservation\_id, price).
- **Active 90-Day Partition:** $3.8 \times 10^9 \times (90/365) \times 64 \text{ B} \approx 60 \text{ GB RAM}$. Fits comfortably in database buffer pools.

| Component | AWS Instance Type / Topology | Specifications | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Edge API Gateway** | 20 $\times$ `c6i.2xlarge` (EKS) | 8 vCPU, 16 GB RAM, 12.5 Gbps | Terminates TLS, rate limits reservation requests, enforces idempotency keys |
| **Availability Search Cache** | 12 $\times$ `r6i.2xlarge` (Redis Cluster) | 8 vCPU, 64 GB RAM per node | Read-only calendar availability cache; updated via Kafka CDC event stream |
| **Saga Orchestrators** | 16 $\times$ `c6i.2xlarge` (Temporal Workers) | 8 vCPU, 16 GB RAM | Coordinates multi-step booking sagas (Lock, Payment, Confirm, Compensate) |
| **Inventory & Reservation DB**| 1 Primary + 3 Read Replicas | `db.r6i.8xlarge` (Aurora PostgreSQL) | 256 GB RAM, 20,000 Provisioned IOPS (io2), monthly partitioned inventory |
| **Kafka Event Stream** | 6 $\times$ `i3en.xlarge` (KRaft) | 4 vCPU, 32 GB RAM, NVMe | Broadcasts `ReservationConfirmed` and `InventoryUpdated` events to read caches |

#### Visual Architecture Blueprint
![Figure 17.14: Distributed Hotel & Flight Booking Inventory System](visuals/arch_booking_inventory.png){width=95%}

#### Architectural Workflow & Mechanics
1. **Search vs. Reservation Flow Separation (CQRS):**
   - Search queries (accounting for $> 99\%$ of traffic) hit read replicas and Redis availability caches.
   - Reservation transactions bypass the cache and hit the primary database directly to ensure strict ACID isolation.
2. **Reservation Saga Orchestration & Lock Minimization:**
   - Holding row-level database locks across an external payment gateway call (which takes 2–30 seconds) blocks all concurrent transactions and exhausts the connection pool.
   - The booking flow executes as a distributed **Three-Transaction Saga**:
     - **Tx1 (Lock, Reserve & Release):** Opens local DB transaction. Uses `SELECT ... FOR UPDATE` to increment `booked_units`, inserts reservation as `PENDING`, and **COMMITs immediately**, releasing DB locks in $< 10\text{ms}$.
     - **External Step (Payment):** Executes credit card authorization via payment provider. Zero database resources are held.
     - **Tx2 (Confirm on Success):** Updates reservation status from `PENDING` to `CONFIRMED`.
     - **Tx3 (Compensate on Failure):** If payment times out or card is declined, runs compensating transaction: `UPDATE inventory_calendar SET booked_units = booked_units - 1` and marks reservation `FAILED`.
3. **Inventory Bucket Partitioning (Flash-Sale Contention):**
   - For high-demand flash sales (e.g., concert tickets or popular resort inventory where thousands of users contend for the same date), a single row lock becomes an extreme bottleneck.
   - **Bucket Partitioning** splits the available units across $K$ sub-buckets (e.g., 20 buckets of 25 units):
     $$\text{bucket\_id} = \text{random}(1, K)$$

   - Booking requests lock independent sub-buckets in parallel, multiplying concurrent write throughput by $K \times$.
4. **Canonical Lock Ordering (Deadlock Elimination):**
   - When a user reserves a 5-night stay (e.g., Sept 1 to Sept 5), concurrent transactions booking overlapping dates can trigger circular deadlocks.
   - The system enforces a strict canonical locking order: calendar rows must **always** be locked in ascending chronological order:
     ```sql
     SELECT * FROM inventory_calendar 
     WHERE property_id = ? AND calendar_date BETWEEN ? AND ?
     ORDER BY calendar_date ASC 
     FOR UPDATE;
     ```

   - This mathematical total order ($\prec$) eliminates the circular wait condition (Coffman condition 4), making deadlocks mathematically impossible.

#### API Contracts & Interface Specs
```json
// GET /v1/availability?property_id=htl_42&check_in=2026-09-01&check_out=2026-09-05&guests=2
Response (200 OK):
{
  "property_id": "htl_42",
  "available_rooms": [
    {
      "room_type": "DELUXE_KING",
      "units_available": 3,
      "price_per_night_cents": 25000,
      "cancellation_policy": "FREE_48H"
    }
  ]
}
// POST /v1/reservations (Create Reservation)
Header: X-Idempotency-Key: "res_usr42_htl42_20260901"
{
  "user_id": "usr_42",
  "property_id": "htl_42",
  "room_type": "DELUXE_KING",
  "check_in": "2026-09-01",
  "check_out": "2026-09-05",
  "payment_token": "tok_visa_8812"
}
// Response (201 Created):
{
  "reservation_id": "rsv_f81d4",
  "status": "CONFIRMED",
  "total_cents": 100000,
  "cancellation_deadline": "2026-08-30T00:00:00Z"
}
```

#### Production Database Schema & Data Model (PostgreSQL DDL)
```sql
CREATE TABLE properties (
    property_id UUID PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    location_lat DOUBLE PRECISION,
    location_lon DOUBLE PRECISION,
    total_rooms INT NOT NULL
);

-- Partitioned inventory calendar table
CREATE TABLE inventory_calendar (
    property_id UUID NOT NULL REFERENCES properties(property_id),
    room_type VARCHAR(32) NOT NULL,
    calendar_date DATE NOT NULL,
    bucket_id INT NOT NULL DEFAULT 1,
    total_units INT NOT NULL,
    booked_units INT NOT NULL DEFAULT 0,
    price_per_night_cents INT NOT NULL,
    PRIMARY KEY (property_id, room_type, calendar_date, bucket_id),
    CONSTRAINT chk_units_capacity CHECK (booked_units <= total_units)
) PARTITION BY RANGE (calendar_date);

-- Monthly inventory partitions
CREATE TABLE inventory_calendar_2026_09 PARTITION OF inventory_calendar
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');

CREATE TABLE reservations (
    reservation_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    property_id UUID NOT NULL REFERENCES properties(property_id),
    room_type VARCHAR(32) NOT NULL,
    check_in DATE NOT NULL,
    check_out DATE NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('PENDING','CONFIRMED','CANCELLED','FAILED')),
    total_cents INT NOT NULL,
    idempotency_key VARCHAR(128) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reservations_user ON reservations(user_id, created_at DESC);
CREATE INDEX idx_reservations_property ON reservations(property_id, check_in);
CREATE INDEX idx_reservations_reaper ON reservations(expires_at) WHERE status = 'PENDING';
```

#### Step-by-Step Execution Sequence
1. **Search Query:** Search Service queries Redis availability cache. If cache misses, reads from PostgreSQL read replica. Returns room types and prices in $< 35\text{ms}$.
2. **Tx1 (Reserve Inventory):** User initiates booking. Saga Orchestrator starts local DB transaction:
   - Queries `inventory_calendar` with `ORDER BY calendar_date ASC FOR UPDATE`.
   - Verifies `booked_units < total_units` for each night.
   - Increments `booked_units = booked_units + 1`.
   - Inserts reservation as `PENDING` with 10-minute expiration (`expires_at = now() + interval '10 minutes'`).
   - Commits transaction and releases all DB connections in $< 8\text{ms}$.
3. **External Payment Call:** Orchestrator calls Payment Gateway. Database is completely unburdened.
4. **Tx2 (Confirm) or Tx3 (Compensate):**
   - On payment success: Tx2 marks reservation `CONFIRMED`. Emits `ReservationConfirmed` event to Kafka to invalidate Redis cache.
   - On payment failure or timeout: Tx3 decrements `booked_units` back and marks reservation `FAILED`.
5. **Background Reaper Safety Net:** Cron worker runs every 60 seconds: `SELECT * FROM reservations WHERE status = 'PENDING' AND expires_at < now() FOR UPDATE SKIP LOCKED`. For each expired reservation, rolls back inventory and transitions state to `CANCELLED`.

#### Production Failure Modes & Operational Mitigations

| Failure Scenario | Catastrophic Risk | Architectural Mitigation & Recovery Protocol |
| :--- | :--- | :--- |
| **Payment Gateway Indeterminate Timeout** | Credit card charge request times out after 15 seconds; system cannot tell if user was charged. | **Idempotent Reconciliation Poller:** Orchestrator enters `PAYMENT_PENDING` state. Worker queries payment gateway status endpoint with `idempotency_key` before deciding whether to execute Tx2 (Confirm) or Tx3 (Compensate). |
| **Abandoned Cart Seat Holding** | Malicious bot opens 5,000 checkout sessions, locking all inventory without paying. | **Strict 10-Minute Lease with Automated Reaper:** Reservations in `PENDING` status expire after 10 minutes. A background reaper executes compensating transactions to restore inventory to the pool. |
| **Multi-Night Deadlock Hazard** | User A books Sept 1–5 while User B books Sept 5–1, resulting in database deadlocks. | **Canonical Ascending Date Locking:** All multi-row inventory transactions enforce strict `ORDER BY calendar_date ASC FOR UPDATE`. Total ordering mathematically guarantees zero circular wait deadlocks. |
| **Flash Mob Ticket Release Stampede** | 100,000 users click "Buy" on the same venue simultaneously, causing extreme row-lock contention. | **Inventory Bucket Partitioning & Virtual Waiting Room:** Inventory is divided into 50 sub-buckets, allowing 50 concurrent row locks. A Cloudflare/Envoy Virtual Waiting Room meters client checkout rates. |

#### Staff-Level Interview Verbalization
> *"Our booking engine guarantees zero double-bookings by establishing PostgreSQL as the single source of truth under strict ACID isolation. To support high concurrent throughput without exhausting database connections, we structure booking as a Three-Transaction Saga: Tx1 acquires row locks in canonical chronological order, increments booked units, and immediately commits in under 10 milliseconds. The payment call executes outside any database transaction. On success, Tx2 confirms the reservation; on failure, Tx3 executes a compensating decrement. We eliminate deadlock hazards through ascending date locking, scale hot-spot flash sales using inventory bucket partitioning, and recover abandoned reservations via an automated background reaper."*
