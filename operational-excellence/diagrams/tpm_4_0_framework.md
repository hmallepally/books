graph TD
    subgraph "TPM 4.0 Framework"
        subgraph "Traditional TPM Pillars"
            P1[Autonomous Maintenance]
            P2[Planned Maintenance]
            P3[Quality Maintenance]
            P4[Focused Improvement]
            P5[Early Equipment Management]
            P6[Training & Education]
            P7[Safety & Environment]
            P8[Office TPM]
        end
        
        subgraph "AI Enhancement Layer"
            AI1[Predictive Analytics]
            AI2[Computer Vision]
            AI3[IoT Sensors]
            AI4[Machine Learning]
            AI5[Digital Twins]
            AI6[Automated Alerts]
        end
        
        subgraph "Business Outcomes"
            O1[Zero Breakdowns]
            O2[Zero Defects]
            O3[Zero Accidents]
            O4[Maximum Efficiency]
            O5[Cost Optimization]
            O6[Continuous Improvement]
        end
    end
    
    P1 --> AI1
    P2 --> AI2
    P3 --> AI3
    P4 --> AI4
    P5 --> AI5
    P6 --> AI6
    P7 --> AI1
    P8 --> AI4
    
    AI1 --> O1
    AI2 --> O2
    AI3 --> O3
    AI4 --> O4
    AI5 --> O5
    AI6 --> O6
    
    style P1 fill:#e8f5e8
    style P2 fill:#e8f5e8
    style P3 fill:#e8f5e8
    style P4 fill:#e8f5e8
    style P5 fill:#e8f5e8
    style P6 fill:#e8f5e8
    style P7 fill:#e8f5e8
    style P8 fill:#e8f5e8
    
    style AI1 fill:#e1f5fe
    style AI2 fill:#e1f5fe
    style AI3 fill:#e1f5fe
    style AI4 fill:#e1f5fe
    style AI5 fill:#e1f5fe
    style AI6 fill:#e1f5fe
    
    style O1 fill:#fff3e0
    style O2 fill:#fff3e0
    style O3 fill:#fff3e0
    style O4 fill:#fff3e0
    style O5 fill:#fff3e0
    style O6 fill:#fff3e0
