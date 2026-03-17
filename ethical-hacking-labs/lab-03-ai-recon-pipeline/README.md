# Lab 3: AI Reconnaissance Pipeline

**Book Reference**: Chapter 3 — *Reconnaissance: AI Sees What You Don't*

---

## Purpose

Build an automated reconnaissance pipeline that combines traditional OSINT techniques with AI-powered analysis. This lab teaches you how ethical hackers gather intelligence about a target before ever touching a vulnerability scanner.

## What You'll Learn

1. **Subdomain Enumeration** — Discover subdomains via Certificate Transparency logs (crt.sh)
2. **Technology Fingerprinting** — Identify web frameworks, servers, and CMS from HTTP headers
3. **AI Analysis** — Use an LLM to synthesize findings and recommend attack vectors
4. **Report Generation** — Output structured JSON reconnaissance reports

## Pipeline Architecture

```
  Target Domain
       │
       ▼
  ┌─────────────┐
  │  Stage 1:   │   crt.sh Certificate Transparency API
  │  Subdomains │   → Discover all subdomains
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Stage 2:   │   HTTP headers + response analysis
  │  Tech Stack │   → Identify frameworks, servers, CMS
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Stage 3:   │   LLM prompt with all gathered data
  │  AI Analysis│   → Attack vectors, risk assessment
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Stage 4:   │   Structured JSON output
  │  Report     │   → Actionable recon report
  └─────────────┘
```

## Requirements

```
pip install requests beautifulsoup4
```

## Usage

```bash
# Run against a domain you own or have authorization to test
python recon_pipeline.py --target yourdomain.com

# Output: recon_report.json
```

## ⚠️ Legal Notice

**Only test against domains you own or have explicit written authorization to test.**
Unauthorized reconnaissance is illegal in most jurisdictions.

## Key Takeaway

> *"Chen's forty-percent rule: spend 40% of engagement time on reconnaissance. AI doesn't change that rule — it changes the speed at which you satisfy it."*
