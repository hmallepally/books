graph TB
    subgraph "Business Applications"
        BA1[Predictive Analytics]
        BA2[Process Optimization]
        BA3[Quality Control]
        BA4[Customer Insights]
    end
    
    subgraph "AI/ML Layer"
        AI1[Machine Learning Models]
        AI2[Deep Learning Networks]
        AI3[Natural Language Processing]
        AI4[Computer Vision]
        AI5[Robotic Process Automation]
    end
    
    subgraph "Data Processing"
        DP1[Data Preprocessing]
        DP2[Feature Engineering]
        DP3[Model Training]
        DP4[Real-time Processing]
    end
    
    subgraph "Data Sources"
        DS1[IoT Sensors]
        DS2[Production Systems]
        DS3[Customer Data]
        DS4[External APIs]
        DS5[Historical Records]
    end
    
    subgraph "Infrastructure"
        INF1[Cloud Computing]
        INF2[Edge Computing]
        INF3[Data Storage]
        INF4[Security & Compliance]
    end
    
    DS1 --> DP1
    DS2 --> DP1
    DS3 --> DP1
    DS4 --> DP1
    DS5 --> DP1
    
    DP1 --> DP2
    DP2 --> DP3
    DP3 --> AI1
    DP3 --> AI2
    DP3 --> AI3
    DP3 --> AI4
    DP3 --> AI5
    
    AI1 --> DP4
    AI2 --> DP4
    AI3 --> DP4
    AI4 --> DP4
    AI5 --> DP4
    
    DP4 --> BA1
    DP4 --> BA2
    DP4 --> BA3
    DP4 --> BA4
    
    INF1 --> DP1
    INF2 --> DP1
    INF3 --> DP1
    INF4 --> DP1
    
    style BA1 fill:#e8f5e8
    style BA2 fill:#e8f5e8
    style BA3 fill:#e8f5e8
    style BA4 fill:#e8f5e8
    
    style AI1 fill:#e1f5fe
    style AI2 fill:#e1f5fe
    style AI3 fill:#e1f5fe
    style AI4 fill:#e1f5fe
    style AI5 fill:#e1f5fe
    
    style DP1 fill:#f3e5f5
    style DP2 fill:#f3e5f5
    style DP3 fill:#f3e5f5
    style DP4 fill:#f3e5f5
    
    style DS1 fill:#fff3e0
    style DS2 fill:#fff3e0
    style DS3 fill:#fff3e0
    style DS4 fill:#fff3e0
    style DS5 fill:#fff3e0
    
    style INF1 fill:#fce4ec
    style INF2 fill:#fce4ec
    style INF3 fill:#fce4ec
    style INF4 fill:#fce4ec
