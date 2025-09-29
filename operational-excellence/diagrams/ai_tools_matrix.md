graph LR
    subgraph "Operational Functions"
        F1[Quality Control]
        F2[Production Planning]
        F3[Maintenance]
        F4[Supply Chain]
        F5[Customer Service]
        F6[Process Optimization]
    end
    
    subgraph "AI Tools & Technologies"
        T1[Machine Learning]
        T2[Computer Vision]
        T3[Natural Language Processing]
        T4[Robotic Process Automation]
        T5[IoT & Sensors]
        T6[Predictive Analytics]
        T7[Digital Twins]
        T8[Chatbots & Virtual Assistants]
    end
    
    F1 --> T1
    F1 --> T2
    F1 --> T6
    
    F2 --> T1
    F2 --> T6
    F2 --> T7
    
    F3 --> T1
    F3 --> T5
    F3 --> T6
    F3 --> T7
    
    F4 --> T1
    F4 --> T4
    F4 --> T6
    F4 --> T7
    
    F5 --> T3
    F5 --> T4
    F5 --> T8
    
    F6 --> T1
    F6 --> T4
    F6 --> T6
    F6 --> T7
    
    style F1 fill:#e8f5e8
    style F2 fill:#e8f5e8
    style F3 fill:#e8f5e8
    style F4 fill:#e8f5e8
    style F5 fill:#e8f5e8
    style F6 fill:#e8f5e8
    
    style T1 fill:#e1f5fe
    style T2 fill:#e1f5fe
    style T3 fill:#e1f5fe
    style T4 fill:#e1f5fe
    style T5 fill:#e1f5fe
    style T6 fill:#e1f5fe
    style T7 fill:#e1f5fe
    style T8 fill:#e1f5fe
