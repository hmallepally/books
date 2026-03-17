"""
Lab 4.1 — AI-Assisted Vulnerability Scanner
Book: The Ethical Hacker's Playbook, Chapter 4

Scans a target web application for common vulnerabilities,
then generates an LLM prompt to identify attack chains.

IMPORTANT: Only test against systems you own or have authorization to test.
Recommended targets: OWASP Juice Shop, DVWA.

Usage:
    python vuln_scanner.py --target http://localhost:3000
"""

import argparse
import json
from datetime import datetime

import requests


# ──────────────────────────────────────────────────────────
# Check 1: Security Headers
# ──────────────────────────────────────────────────────────

def check_security_headers(base_url):
    """
    Check for missing HTTP security headers.
    
    Modern web applications should include headers that instruct the browser
    to enable security features. Missing headers = missing defenses.
    
    Learning: These are quick wins in any penetration test. Missing headers
    are easy to find and easy to fix, but they're often overlooked because
    automated scanners flag them as "informational" or "low."
    """
    findings = []
    
    required_headers = {
        "Content-Security-Policy": {
            "severity": "Medium",
            "description": "CSP prevents XSS by restricting script sources",
            "impact": "Without CSP, the application is more vulnerable to "
                      "cross-site scripting (XSS) attacks",
        },
        "Strict-Transport-Security": {
            "severity": "Medium",
            "description": "HSTS forces HTTPS and prevents downgrade attacks",
            "impact": "Users could be redirected to HTTP, enabling "
                      "man-in-the-middle attacks",
        },
        "X-Content-Type-Options": {
            "severity": "Low",
            "description": "Prevents MIME-type sniffing",
            "impact": "Browsers might interpret files as executable scripts",
        },
        "X-Frame-Options": {
            "severity": "Medium",
            "description": "Prevents clickjacking via iframe embedding",
            "impact": "Attackers could overlay invisible frames over the UI "
                      "to hijack user clicks",
        },
        "Referrer-Policy": {
            "severity": "Low",
            "description": "Controls what referrer info is sent with requests",
            "impact": "Sensitive URL parameters could leak to third parties",
        },
    }
    
    try:
        resp = requests.get(base_url, timeout=10)
        
        for header, info in required_headers.items():
            if header.lower() not in {h.lower() for h in resp.headers}:
                findings.append({
                    "type": "Missing Security Header",
                    "header": header,
                    "severity": info["severity"],
                    "description": info["description"],
                    "impact": info["impact"],
                })
    except requests.RequestException as e:
        findings.append({
            "type": "Connection Error",
            "severity": "Info",
            "description": f"Could not connect to {base_url}: {e}",
        })
    
    return findings


# ──────────────────────────────────────────────────────────
# Check 2: Information Disclosure
# ──────────────────────────────────────────────────────────

def check_information_disclosure(base_url):
    """
    Check for information leakage that aids attackers.
    
    Servers often reveal version numbers, framework details, and debug
    information that helps attackers choose specific exploits.
    
    Learning: Information disclosure is the bridge between reconnaissance
    and exploitation. Every version number narrows the attacker's search
    from "which vulnerability?" to "this exact CVE."
    """
    findings = []
    
    try:
        resp = requests.get(base_url, timeout=10)
        
        # Check Server header for version info
        server = resp.headers.get("Server", "")
        if server and any(char.isdigit() for char in server):
            findings.append({
                "type": "Server Version Disclosed",
                "severity": "Low",
                "detail": f"Server header reveals: {server}",
                "impact": "Version info helps attackers identify specific CVEs",
            })
        
        # Check X-Powered-By
        powered_by = resp.headers.get("X-Powered-By", "")
        if powered_by:
            findings.append({
                "type": "Technology Stack Disclosed",
                "severity": "Low",
                "detail": f"X-Powered-By: {powered_by}",
                "impact": "Framework info narrows exploit selection",
            })
        
        # Check common debug/info endpoints
        debug_paths = [
            "/api", "/api/v1", "/swagger", "/docs", "/graphql",
            "/debug", "/health", "/status", "/info", "/env",
            "/.env", "/robots.txt", "/sitemap.xml", "/.git/config",
        ]
        
        for path in debug_paths:
            try:
                r = requests.get(f"{base_url}{path}", timeout=5, 
                               allow_redirects=False)
                if r.status_code == 200 and len(r.text) > 50:
                    findings.append({
                        "type": "Exposed Endpoint",
                        "severity": "Medium" if path in (
                            "/.env", "/.git/config", "/debug", "/env"
                        ) else "Info",
                        "detail": f"{path} returned {r.status_code} "
                                  f"({len(r.text)} bytes)",
                        "impact": "Exposed endpoints may reveal internal "
                                  "configuration or API structure",
                    })
            except requests.RequestException:
                pass
    
    except requests.RequestException as e:
        findings.append({
            "type": "Connection Error",
            "severity": "Info",
            "description": str(e),
        })
    
    return findings


# ──────────────────────────────────────────────────────────
# Check 3: IDOR (Insecure Direct Object Reference)
# ──────────────────────────────────────────────────────────

def check_idor(base_url):
    """
    Test for basic IDOR / Broken Object Level Authorization (BOLA).
    
    IDOR occurs when an application exposes internal object references
    (e.g., user IDs in URL) without verifying the requester's authorization.
    
    BOLA is the #1 vulnerability in the OWASP API Security Top 10 — and
    it's responsible for ~40% of all API breaches.
    
    Learning: AI-assisted testing can generate thousands of IDOR test cases
    by understanding the API's naming conventions and data patterns. A human
    tester might try IDs 1-10. An AI-guided fuzzer tries IDs based on
    patterns it observes in the application's responses.
    """
    findings = []
    
    # Common patterns for IDOR testing
    idor_patterns = [
        "/api/Users/{id}",
        "/api/users/{id}",
        "/api/v1/users/{id}",
        "/rest/user/{id}",
        "/api/Products/{id}",
    ]
    
    for pattern in idor_patterns:
        for test_id in [1, 2, 100]:
            url = f"{base_url}{pattern.format(id=test_id)}"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        # Check if we got user data without authentication
                        sensitive_keys = {"email", "password", "ssn", 
                                        "credit", "token", "secret"}
                        exposed = sensitive_keys & {
                            k.lower() for k in _flatten_keys(data)
                        }
                        
                        severity = "High" if exposed else "Medium"
                        findings.append({
                            "type": "Potential IDOR",
                            "severity": severity,
                            "url": url,
                            "status": resp.status_code,
                            "detail": f"Accessible without auth. "
                                     f"Sensitive fields: {exposed or 'none detected'}",
                            "impact": "Unauthenticated access to user/object data"
                                     " via direct reference",
                        })
                        break  # Found one — no need to test more IDs
                    except (json.JSONDecodeError, Exception):
                        pass
            except requests.RequestException:
                pass
    
    return findings


def _flatten_keys(obj, prefix=""):
    """Recursively extract all keys from a nested dict/list."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _flatten_keys(v, f"{prefix}{k}.")
    elif isinstance(obj, list) and obj:
        keys |= _flatten_keys(obj[0], prefix)
    return keys


# ──────────────────────────────────────────────────────────
# AI Chain Analysis
# ──────────────────────────────────────────────────────────

def generate_chain_analysis_prompt(target, all_findings):
    """
    Generate an LLM prompt that asks AI to identify attack chains.
    
    Individual findings are useful. But the real value of a penetration test
    is showing how individual weaknesses CHAIN together into exploitable paths.
    
    Example: Missing CSP + IDOR + No rate limiting = 
    "An attacker can enumerate all user accounts, inject scripts via XSS,
    and exfiltrate session tokens — all without authentication."
    
    Learning: This is the difference between a vulnerability report (list)
    and a risk assessment (story). Executives don't act on lists.
    They act on stories.
    """
    findings_text = json.dumps(all_findings, indent=2)
    
    prompt = f"""You are an expert penetration tester analyzing findings from 
an authorized security assessment. Your job is to identify ATTACK CHAINS — 
combinations of individual findings that together create exploitable paths.

TARGET: {target}
DATE: {datetime.now().strftime('%Y-%m-%d')}

INDIVIDUAL FINDINGS ({len(all_findings)} total):
{findings_text}

ANALYSIS REQUESTED:

1. **Attack Chain Analysis**: Identify how 2+ findings could be combined 
   into realistic attack scenarios. For each chain:
   - Name the chain (e.g., "Unauthenticated Data Exfiltration")
   - List the findings that compose it
   - Describe the step-by-step attack path
   - Rate the combined risk (Critical/High/Medium/Low)
   - Estimate exploitability (Easy/Moderate/Hard)

2. **Quick Wins**: Which findings can be fixed immediately with minimal effort?

3. **Priority Remediation**: Rank the top 3 fixes that would break the most 
   attack chains.

4. **Executive Summary**: Write a 3-sentence summary suitable for a board 
   presentation.

Remember: Individual findings tell you what's broken. Chains tell you 
what an attacker can DO. Focus on the chains."""

    return prompt


# ──────────────────────────────────────────────────────────
# Main Scanner
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI-Assisted Vulnerability Scanner — Lab 4.1"
    )
    parser.add_argument(
        "--target", required=True,
        help="Target URL (e.g., http://localhost:3000)"
    )
    parser.add_argument(
        "--output", default="vuln_report.json",
        help="Output file (default: vuln_report.json)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🛡️  AI-Assisted Vulnerability Scanner — Lab 4.1")
    print("   The Ethical Hacker's Playbook")
    print("=" * 60)
    print(f"\n⚠️  AUTHORIZED TESTING ONLY — Target: {args.target}")

    all_findings = []

    # Check 1: Security Headers
    print(f"\n[Check 1/3] Security Headers...")
    headers_findings = check_security_headers(args.target)
    all_findings.extend(headers_findings)
    print(f"  Found {len(headers_findings)} issue(s)")

    # Check 2: Information Disclosure
    print(f"\n[Check 2/3] Information Disclosure...")
    info_findings = check_information_disclosure(args.target)
    all_findings.extend(info_findings)
    print(f"  Found {len(info_findings)} issue(s)")

    # Check 3: IDOR
    print(f"\n[Check 3/3] IDOR / Broken Authorization...")
    idor_findings = check_idor(args.target)
    all_findings.extend(idor_findings)
    print(f"  Found {len(idor_findings)} issue(s)")

    # Summary
    print(f"\n{'─' * 60}")
    print(f"📊 Total findings: {len(all_findings)}")
    by_severity = {}
    for f in all_findings:
        sev = f.get("severity", "Info")
        by_severity[sev] = by_severity.get(sev, 0) + 1
    for sev in ["Critical", "High", "Medium", "Low", "Info"]:
        if sev in by_severity:
            print(f"   {sev}: {by_severity[sev]}")
    print(f"{'─' * 60}")

    # AI Chain Analysis
    print(f"\n[AI] Generating chain analysis prompt...")
    ai_prompt = generate_chain_analysis_prompt(args.target, all_findings)
    print(f"\n{'─' * 60}")
    print("📋 Copy the prompt below into ChatGPT or Claude:")
    print(f"{'─' * 60}")
    print(ai_prompt)
    print(f"{'─' * 60}\n")

    # Save report
    report = {
        "meta": {
            "target": args.target,
            "timestamp": datetime.now().isoformat(),
            "tool": "AI Vuln Scanner — Ethical Hacker's Playbook Lab 4.1",
        },
        "findings": all_findings,
        "ai_chain_prompt": ai_prompt,
    }
    
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
