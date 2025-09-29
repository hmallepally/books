graph TB
    subgraph "Data Sources"
        A[IoT Sensors] --> D
        B[Production Systems] --> D
        C[Quality Reports] --> D
    end
    
    subgraph "Data Management"
        D[Data Collection] --> E[Data Validation]
        E --> F[Data Storage]
    end
    
    subgraph "AI Processing"
        F --> G[Data Preprocessing]
        G --> H[Machine Learning Models]
        H --> I[Prediction Engine]
    end
    
    subgraph "Output & Actions"
        I --> J[Real-time Alerts]
        I --> K[Quality Reports]
        I --> L[Automated Actions]
    end
    
    subgraph "User Interface"
        J --> M[Dashboard]
        K --> M
        L --> M
    end
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#e8f5e8
    style H fill:#e8f5e8
    style I fill:#e8f5e8
    style J fill:#fff3e0
    style K fill:#fff3e0
    style L fill:#fff3e0
    style M fill:#fce4ec
