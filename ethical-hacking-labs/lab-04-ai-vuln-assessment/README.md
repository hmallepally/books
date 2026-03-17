# Lab 4: AI-Assisted Vulnerability Scanner

**Book Reference**: Chapter 4 — *Vulnerability Assessment: The AI Vulnerability Hunter*

---

## Purpose

Build an AI-assisted vulnerability scanner that checks for common web security issues, then uses an LLM to identify **attack chains** — combinations of individual findings that together create exploitable paths.

## What You'll Learn

1. **Security Header Analysis** — Check for missing protective HTTP headers
2. **Information Disclosure** — Detect verbose error pages, server versions, debug endpoints
3. **IDOR Testing** — Test for Insecure Direct Object Reference (broken authorization)
4. **AI Chain Analysis** — Use an LLM to connect individual findings into attack paths

## The Key Insight

> *"The scanner finds 5 individual issues. The AI identifies how they chain into 2 exploitable attack paths. That's the difference between vulnerability scanning and vulnerability assessment."*

## Architecture

```
  Target URL
       │
   ┌───┴───┐
   │ Check │  Security headers (CSP, HSTS, X-Frame-Options)
   │ Check │  Information disclosure (Server header, error pages)
   │ Check │  IDOR / authorization (sequential ID access)
   └───┬───┘
       │
       ▼
  ┌─────────────┐
  │  AI Engine  │  LLM analyzes ALL findings together
  │  (Prompt)   │  → Identifies attack chains
  └──────┬──────┘
         │
         ▼
  Structured Report
  (Individual findings + chained attack paths)
```

## Requirements

```
pip install requests
```

**Target Application** (pick one):
```bash
# Option A: OWASP Juice Shop (recommended)
docker run -p 3000:3000 bkimminich/juice-shop

# Option B: DVWA (Damn Vulnerable Web Application)
docker run -d -p 80:80 vulnerables/web-dvwa
```

## Usage

```bash
# Run against your local vulnerable app
python vuln_scanner.py --target http://localhost:3000

# Output: vuln_report.json + AI analysis prompt
```

## ⚠️ Legal Notice

**Test ONLY against intentionally vulnerable applications (Juice Shop, DVWA) or systems you have written authorization to test.**
