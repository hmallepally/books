import os

base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'

def insert_after(file_path, search_str, text_to_insert):
    full_path = os.path.join(base_dir, file_path)
    if not os.path.exists(full_path):
        print('Missing file:', full_path)
        return
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if search_str in line:
            lines.insert(i+1, '\n' + text_to_insert + '\n')
            with open(full_path, 'w', encoding='utf-8') as fw:
                fw.writelines(lines)
            print('Inserted in', file_path)
            return
    print('Failed to find string in', file_path, '->', search_str)

# Round 4
insert_after('02-problem-decomposition/base.md', 'map directly to known patterns.', '![Problem Decomposition Tree — Breaking Complex Problems into Sub-Problems](visuals/decomposition_tree.jpg){width=85%}')
insert_after('06-functional-streams/base.md', 'lazy execution model.', '![Lazy Evaluation and Short-Circuiting in Streams](visuals/lazy_evaluation.jpg){width=85%}')
insert_after('08-concurrency-performance/base.md', '| **Deadlock Risk**', '![Database Deadlock Cycle — Circular Wait Conditions](visuals/deadlock_diagram.jpg){width=85%}')
insert_after('09-algorithms-assessment/base.md', '| $O(N!)$ |', '![Big-O Time Complexity Comparison Graph](visuals/big_o_comparison.jpg){width=85%}')
insert_after('14-mastering-decomposition/base.md', 'systematic diagnostic process.', '![Problem Analysis Canvas — Structured Decomposition Framework](visuals/problem_analysis_canvas.jpg){width=85%}')
insert_after('16-system-architecture/base.md', '**AP:** The system remains available', '![CAP Theorem — Consistency, Availability, and Partition Tolerance Trade-offs](visuals/cap_theorem.jpg){width=85%}')
insert_after('16-system-architecture/base.md', '![Monolithic vs Microservices', '![System Evolution — Scaling from Monolith to Microservices](visuals/system_evolution.jpg){width=85%}')
insert_after('21-message-brokers/base.md', 'This prevents duplicate processing of messages.', '![Kafka Partitions and Consumer Group Parallelism](visuals/kafka_partitions.jpg){width=85%}')

# Round 5
insert_after('01-invariant-first/base.md', 'guess the boundary updates', '![Loop Invariant States — Boundary Contraction in Binary Search](visuals/loop_invariant_states.jpg){width=85%}')
insert_after('02-problem-decomposition/base.md', 'eliminates 50% of wrong algorithm choices', '![Constraint-to-Complexity Flowchart](visuals/constraint_flowchart.jpg){width=85%}')
insert_after('03-case-studies/base.md', 'execute matches in real time.', '![ZenithTrade High-Frequency Matching Engine Architecture](visuals/zenithtrade_architecture.jpg){width=85%}')
insert_after('03-case-studies/base.md', 'establish consensus-based recovery.', '![ChiramTrust Decentralized Identity Wallet Architecture](visuals/chiramtrust_architecture.jpg){width=85%}')
insert_after('04-oop-principles/base.md', 'violating the core safety boundaries', '![God Object Violation Detector — Single Responsibility Principle](visuals/oop_violation_detector.jpg){width=85%}')
insert_after('08-concurrency-performance/base.md', 'assigns the carrier thread to another task.', '![Thread Lifecycle and Context Switching States](visuals/thread_lifecycle.jpg){width=85%}')
insert_after('14-mastering-decomposition/base.md', 'The Pattern Recognition Decision Tree (Expanded)', '![Pattern Selection Decision Matrix](visuals/decomposition_decision.jpg){width=85%}')
insert_after('15-mock-assessment-sets/base.md', 'knowledge.', '![Assessment Pacing Strategy and Time Allocation](visuals/pacing_strategy.jpg){width=85%}')
insert_after('16-system-architecture/base.md', '### Consistent Hashing for Instrument Sharding', '![Consistent Hashing Ring — Distributed Key Routing](visuals/consistent_hashing.jpg){width=85%}')
insert_after('18-database-compliance/base.md', 'track which shard stores a specific partition key.', '![Database Sharding Strategies — Range, Hash, and Directory Based](visuals/sharding_strategies.jpg){width=85%}')
insert_after('22-aiml-llm/base.md', 'Triton Inference Server, or custom gRPC endpoints', '![Model Serving Infrastructure and Real-time Inference](visuals/model_serving.jpg){width=85%}')
