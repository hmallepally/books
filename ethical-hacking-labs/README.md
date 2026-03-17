# 🛡️ The Ethical Hacker's Playbook — Labs

**Companion lab exercises for *The Ethical Hacker's Playbook: AI on Our Side***
*An Evergreen Enterprise Companion by Harinath Mallepally*

---

## About This Repository

This repository contains the hands-on lab exercises referenced throughout the book. Each lab is self-contained with its own README, Python code, and learning objectives.

> ⚠️ **Important**: These tools are for **educational and authorized testing only**. Only test against systems you own or have explicit written authorization to test. Unauthorized testing is illegal.

---

## Labs

| Lab | Chapter | Title | What You Build |
|-----|---------|-------|---------------|
| [Lab 3](./lab-03-ai-recon-pipeline/) | Ch. 3: Reconnaissance | AI Recon Pipeline | Automated subdomain enumeration + technology fingerprinting + AI analysis |
| [Lab 4](./lab-04-ai-vuln-assessment/) | Ch. 4: Vulnerability Assessment | AI-Assisted Vulnerability Scanner | Security header checker + IDOR tester + AI chain analysis |
| [Lab 6](./lab-06-ai-alert-classifier/) | Ch. 6: The Blue Team | AI Alert Classifier | ML-powered SOC alert triage with Random Forest + LLM explanations |
| [Lab 8](./lab-08-home-lab-setup/) | Ch. 8: Career Path | Home Lab Setup Guide | Step-by-step guide to building your own hacking lab |

---

## Prerequisites

### System Requirements
- **Python 3.10+**
- **pip** (Python package manager)
- **Docker** (for Lab 4 — running vulnerable targets)
- **Git**

### Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ethical-hacking-labs.git
cd ethical-hacking-labs

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# or: .venv\Scripts\activate   # Windows

# Install all lab dependencies
pip install -r requirements.txt
```

### Python Dependencies

All labs share a common `requirements.txt`:

```
requests>=2.31.0
beautifulsoup4>=4.12.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
```

---

## Lab Progression

The labs are designed to be completed in order:

```
Lab 3: Reconnaissance    → Gather intelligence (OSINT, subdomains, tech stack)
         ↓
Lab 4: Vulnerability     → Find weaknesses (headers, IDOR, AI chain analysis)
         ↓
Lab 6: Blue Team         → Defend & classify (ML alert triage, LLM explanations)
         ↓
Lab 8: Home Lab          → Build your permanent practice environment
```

---

## ⚖️ Legal & Ethical Guidelines

1. **Authorization**: Never test systems without explicit written permission
2. **Scope**: Stay within the defined scope of any engagement
3. **Data**: Never access, copy, or exfiltrate real sensitive data
4. **Reporting**: Report all findings through proper channels
5. **Lab Targets Only**: Use DVWA, Juice Shop, and other intentionally vulnerable apps for practice

---

## About the Book

*The Ethical Hacker's Playbook: AI on Our Side* is a companion to the Evergreen Enterprise series. It covers:

- **The AI cybersecurity arms race** — $10.5T in cybercrime vs. $212B in defense
- **AI-powered offensive techniques** — reconnaissance, vulnerability hunting, red teaming
- **AI-powered defense** — SOC automation, behavioral analytics, 47-second containment
- **Career paths** — certifications (CEH, OSCP, PNPT), salary data, bug bounty
- **Governance** — EU AI Act, NIST AI RMF, responsible disclosure

Available on Amazon KDP.

---

## License

This code is provided for educational purposes as a companion to the book.
Copyright © 2026 Harinath Mallepally. All rights reserved.
