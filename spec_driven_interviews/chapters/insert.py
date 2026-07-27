import os

base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'

insertions = [
    {
        'file': '02-problem-decomposition/base.md',
        'match': 'solvable sub-problems that map directly to known patterns.',
        'text': '\n![Problem Decomposition Tree — Breaking Complex Problems into Sub-Problems](visuals/decomposition_tree.jpg){width=85%}\n'
    },
    {
        'file': '06-functional-streams/base.md',
        'match': 'To inspect stream internals during test failures, apply these tactics:',
        'text': '\n![Lazy Evaluation and Short-Circuiting in Streams](visuals/lazy_evaluation.jpg){width=85%}\n'
    },
    {
        'file': '08-concurrency-performance/base.md',
        'match': 'High lock contention, database thread starvation, and high risk of deadlocks under load.',
        'text': '\n![Database Deadlock Cycle — Circular Wait Conditions](visuals/deadlock_diagram.jpg){width=85%}\n'
    },
    {
        'file': '08-concurrency-performance/base.md',
        'match': 'Virtual threads are lightweight threads managed by the JVM',
        'text': '\n![Thread Lifecycle and Context Switching States](visuals/thread_lifecycle.jpg){width=85%}\n',
        'before': True
    },
    {
        'file': '09-algorithms-assessment/base.md',
        'match': '| $O(N!)$ | Factorial | Traveling Salesperson | $10$ |',
        'text': '\n![Big-O Time Complexity Comparison Graph](visuals/big_o_comparison.jpg){width=85%}\n'
    },
    {
        'file': '14-mastering-decomposition/base.md',
        'match': 'By rigidly adhering to this canvas, you eliminate the panic of the blank screen and replace it with a systematic diagnostic process.',
        'text': '\n![Problem Analysis Canvas — Structured Decomposition Framework](visuals/problem_analysis_canvas.jpg){width=85%}\n'
    },
    {
        'file': '16-system-architecture/base.md',
        'match': '**AP:** The system remains available but may return stale data (e.g., Cassandra, DynamoDB).',
        'text': '\n![CAP Theorem — Consistency, Availability, and Partition Tolerance Trade-offs](visuals/cap_theorem.jpg){width=85%}\n'
    },
    {
        'file': '16-system-architecture/base.md',
        'match': '![Monolithic vs Microservices vs Event-Driven Architecture](visuals/arch_styles.png){width=80%}',
        'text': '\n![System Evolution — Scaling from Monolith to Microservices](visuals/system_evolution.jpg){width=85%}\n'
    },
    {
        'file': '21-message-brokers/base.md',
        'match': 'A consumer group is a collection of consumers working together to read messages from a topic. Kafka guarantees that each partition is assigned to exactly *one* consumer instance within a consumer group. This prevents duplicate processing of messages.',
        'text': '\n![Kafka Partitions and Consumer Group Parallelism](visuals/kafka_partitions.jpg){width=85%}\n'
    },
    {
        'file': '01-invariant-first/base.md',
        'match': 'Many developers struggle with binary search, often getting trapped in infinite loops or off-by-one errors because they guess the boundary updates',
        'text': '\n![Loop Invariant States — Boundary Contraction in Binary Search](visuals/loop_invariant_states.jpg){width=85%}\n'
    },
    {
        'file': '02-problem-decomposition/base.md',
        'match': 'If N $\\leq$ 10^6, you need O(N). This single rule eliminates 50% of wrong algorithm choices before you write a line of code.',
        'text': '\n![Constraint-to-Complexity Flowchart](visuals/constraint_flowchart.jpg){width=85%}\n'
    },
    {
        'file': '03-case-studies/base.md',
        'match': 'ZenithTrade is a high-frequency, low-latency order matching engine. It is designed to process incoming buy and sell limit orders and execute matches in real time.',
        'text': '\n![ZenithTrade High-Frequency Matching Engine Architecture](visuals/zenithtrade_architecture.jpg){width=85%}\n'
    },
    {
        'file': '03-case-studies/base.md',
        'match': 'ChiramTrust is a decentralized identity wallet that allows users to store credentials locally, negotiate sharing terms with verifiers, and establish consensus-based recovery.',
        'text': '\n![ChiramTrust Decentralized Identity Wallet Architecture](visuals/chiramtrust_architecture.jpg){width=85%}\n'
    },
    {
        'file': '04-oop-principles/base.md',
        'match': 'If your class name ends in `Manager`, `Processor`, or `System`, you have likely built a God Object.',
        'text': '\n![God Object Violation Detector — Single Responsibility Principle](visuals/oop_violation_detector.jpg){width=85%}\n'
    },
    {
        'file': '14-mastering-decomposition/base.md',
        'match': 'The Pattern Recognition Decision Tree (Expanded)',
        'text': '\n![Pattern Selection Decision Matrix](visuals/decomposition_decision.jpg){width=85%}\n'
    },
    {
        'file': '15-mock-assessment-sets/base.md',
        'match': 'For a 45-minute technical assessment, time management is as critical as algorithmic knowledge.',
        'text': '\n![Assessment Pacing Strategy and Time Allocation](visuals/pacing_strategy.jpg){width=85%}\n'
    },
    {
        'file': '16-system-architecture/base.md',
        'match': '### Consistent Hashing for Instrument Sharding',
        'text': '\n![Consistent Hashing Ring — Distributed Key Routing](visuals/consistent_hashing.jpg){width=85%}\n'
    },
    {
        'file': '18-database-compliance/base.md',
        'match': '3. **Directory-Based Sharding:** Utilizing a centralized lookup service (lookup table) to track which shard stores a specific partition key.',
        'text': '\n![Database Sharding Strategies — Range, Hash, and Directory Based](visuals/sharding_strategies.jpg){width=85%}\n'
    },
    {
        'file': '22-aiml-llm/base.md',
        'match': '- **Model Serving:** Deploy models behind low-latency serving infrastructure (TensorFlow Serving, Triton Inference Server, or custom gRPC endpoints).',
        'text': '\n![Model Serving Infrastructure and Real-time Inference](visuals/model_serving.jpg){width=85%}\n'
    }
]

# Ensure we have matches in files
def process():
    for item in insertions:
        path = os.path.join(base_dir, item['file'])
        if not os.path.exists(path):
            print(f"Missing file {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        found = False
        for i, line in enumerate(lines):
            if item['match'] in line:
                if item.get('before'):
                    lines.insert(i, item['text'])
                else:
                    lines.insert(i+1, item['text'])
                found = True
                break
        if not found:
            print(f"Could not find match in {item['file']} for '{item['match']}'")
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"Updated {item['file']}")

process()
