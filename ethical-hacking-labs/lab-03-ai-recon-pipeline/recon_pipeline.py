"""
Lab 3.1 — AI Reconnaissance Pipeline
Book: The Ethical Hacker's Playbook, Chapter 3

Build an automated recon pipeline that discovers subdomains,
fingerprints technologies, and uses AI to analyze findings.

IMPORTANT: Only test against domains you own or have authorization to test.

Usage:
    python recon_pipeline.py --target yourdomain.com
"""

import argparse
import json
import sys
from datetime import datetime

import requests
from bs4 import BeautifulSoup


# ──────────────────────────────────────────────────────────
# Stage 1: Subdomain Enumeration via Certificate Transparency
# ──────────────────────────────────────────────────────────

def enumerate_subdomains(domain):
    """
    Query crt.sh Certificate Transparency logs to discover subdomains.
    
    Certificate Transparency (CT) is a public framework where Certificate 
    Authorities (CAs) must log every SSL/TLS certificate they issue. By 
    querying these logs, we can discover subdomains that may not be visible
    through DNS brute-forcing alone.
    
    Learning: This is one of the most reliable passive recon techniques
    because it uses publicly available data — no packets touch the target.
    """
    print(f"\n[Stage 1] Enumerating subdomains for: {domain}")
    url = f"https://crt.sh/?q=%.{domain}&output=json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract unique subdomain names from certificate records
        certs = response.json()
        subdomains = set()
        for cert in certs:
            name = cert.get("name_value", "")
            # CT logs may contain wildcard entries and newline-separated names
            for sub in name.split("\n"):
                sub = sub.strip().lower()
                if sub and not sub.startswith("*") and domain in sub:
                    subdomains.add(sub)
        
        print(f"  ✅ Found {len(subdomains)} unique subdomains")
        return sorted(subdomains)

    except requests.RequestException as e:
        print(f"  ❌ crt.sh query failed: {e}")
        return [domain]  # Fall back to just the base domain


# ──────────────────────────────────────────────────────────
# Stage 2: Technology Fingerprinting
# ──────────────────────────────────────────────────────────

def fingerprint_technology(subdomain):
    """
    Identify web technologies by analyzing HTTP response headers and HTML content.
    
    Web servers, frameworks, and CMS platforms leave fingerprints in:
    - HTTP headers (Server, X-Powered-By, X-Generator)
    - HTML meta tags and generator tags
    - Cookie names (e.g., JSESSIONID → Java, PHPSESSID → PHP)
    - JavaScript framework signatures
    
    Learning: This is why defensive teams should strip version headers 
    and customize error pages — every piece of version info helps attackers
    narrow their exploit selection.
    """
    result = {
        "subdomain": subdomain,
        "status": None,
        "technologies": [],
        "headers_of_interest": {},
        "title": None,
    }
    
    for scheme in ["https", "http"]:
        url = f"{scheme}://{subdomain}"
        try:
            resp = requests.get(url, timeout=10, allow_redirects=True, 
                              verify=False)  # noqa: S501 - Lab only
            result["status"] = resp.status_code
            
            # Extract technology indicators from headers
            interesting_headers = [
                "Server", "X-Powered-By", "X-Generator", "X-AspNet-Version",
                "X-Frame-Options", "Content-Security-Policy",
                "Strict-Transport-Security", "X-Content-Type-Options"
            ]
            
            for header in interesting_headers:
                value = resp.headers.get(header)
                if value:
                    result["headers_of_interest"][header] = value
                    if header in ("Server", "X-Powered-By", "X-Generator"):
                        result["technologies"].append(f"{header}: {value}")
            
            # Check for missing security headers (a finding in itself)
            security_headers = [
                "Content-Security-Policy", "Strict-Transport-Security",
                "X-Frame-Options", "X-Content-Type-Options"
            ]
            missing = [h for h in security_headers 
                      if h not in resp.headers]
            if missing:
                result["missing_security_headers"] = missing
            
            # Parse HTML for additional clues
            if "text/html" in resp.headers.get("Content-Type", ""):
                soup = BeautifulSoup(resp.text[:10000], "html.parser")
                
                # Page title
                title_tag = soup.find("title")
                if title_tag:
                    result["title"] = title_tag.text.strip()[:100]
                
                # Meta generator tag (WordPress, Drupal, etc.)
                gen = soup.find("meta", attrs={"name": "generator"})
                if gen and gen.get("content"):
                    result["technologies"].append(
                        f"Generator: {gen['content']}"
                    )
            
            break  # Success — don't try the other scheme
            
        except requests.RequestException:
            continue
    
    return result


# ──────────────────────────────────────────────────────────
# Stage 3: AI Analysis Prompt Generation
# ──────────────────────────────────────────────────────────

def generate_ai_analysis_prompt(domain, subdomains, fingerprints):
    """
    Generate a structured LLM prompt for AI-powered analysis.
    
    This prompt is designed to be pasted into ChatGPT, Claude, or any LLM.
    The AI will synthesize all reconnaissance findings and provide:
    - Risk assessment for each subdomain
    - Potential attack vectors based on discovered technologies
    - Recommended next steps for a penetration test
    
    Learning: The quality of AI analysis depends entirely on the quality
    of the prompt. Structured data + clear instructions = better output.
    """
    prompt = f"""You are an expert penetration tester conducting an authorized 
security assessment. Analyze the following reconnaissance data and provide 
a risk assessment.

TARGET: {domain}
DATE: {datetime.now().strftime('%Y-%m-%d')}
SCOPE: Authorized penetration test — all subdomains in scope

DISCOVERED SUBDOMAINS ({len(subdomains)}):
{chr(10).join(f'  - {s}' for s in subdomains[:20])}
{'  ... and ' + str(len(subdomains) - 20) + ' more' if len(subdomains) > 20 else ''}

TECHNOLOGY FINGERPRINTS:
"""
    for fp in fingerprints:
        if fp.get("status"):
            prompt += f"\n  [{fp['subdomain']}] (HTTP {fp['status']})"
            if fp.get("title"):
                prompt += f"\n    Title: {fp['title']}"
            for tech in fp.get("technologies", []):
                prompt += f"\n    Tech: {tech}"
            for header, value in fp.get("headers_of_interest", {}).items():
                prompt += f"\n    Header: {header}: {value}"
            missing = fp.get("missing_security_headers", [])
            if missing:
                prompt += f"\n    ⚠️ Missing: {', '.join(missing)}"

    prompt += """

ANALYSIS REQUESTED:
1. Risk rating (Critical/High/Medium/Low) for each live subdomain
2. Top 5 potential attack vectors based on discovered technologies
3. Recommended next steps for the penetration test
4. Any quick wins (low-effort, high-impact findings)

Format your response as a structured security report."""

    return prompt


# ──────────────────────────────────────────────────────────
# Stage 4: Report Generation
# ──────────────────────────────────────────────────────────

def generate_report(domain, subdomains, fingerprints, ai_prompt):
    """Generate a structured JSON recon report."""
    report = {
        "meta": {
            "target": domain,
            "timestamp": datetime.now().isoformat(),
            "tool": "AI Recon Pipeline — Ethical Hacker's Playbook Lab 3.1",
            "disclaimer": "Authorized testing only"
        },
        "reconnaissance": {
            "subdomains_found": len(subdomains),
            "subdomains": subdomains,
            "live_hosts": [
                fp for fp in fingerprints if fp.get("status")
            ],
            "technologies_detected": list(set(
                tech 
                for fp in fingerprints 
                for tech in fp.get("technologies", [])
            )),
        },
        "ai_analysis_prompt": ai_prompt,
    }
    return report


# ──────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Reconnaissance Pipeline — Lab 3.1"
    )
    parser.add_argument(
        "--target", required=True,
        help="Target domain (must have authorization to test)"
    )
    parser.add_argument(
        "--max-fingerprint", type=int, default=10,
        help="Max subdomains to fingerprint (default: 10)"
    )
    parser.add_argument(
        "--output", default="recon_report.json",
        help="Output file path (default: recon_report.json)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🛡️  AI Reconnaissance Pipeline — Lab 3.1")
    print("   The Ethical Hacker's Playbook")
    print("=" * 60)
    print(f"\n⚠️  AUTHORIZED TESTING ONLY — Target: {args.target}")

    # Stage 1: Subdomain Enumeration
    subdomains = enumerate_subdomains(args.target)

    # Stage 2: Technology Fingerprinting (limit to avoid abuse)
    print(f"\n[Stage 2] Fingerprinting top {args.max_fingerprint} subdomains...")
    fingerprints = []
    for sub in subdomains[:args.max_fingerprint]:
        print(f"  🔍 {sub}...", end=" ")
        fp = fingerprint_technology(sub)
        fingerprints.append(fp)
        status = f"HTTP {fp['status']}" if fp["status"] else "unreachable"
        techs = len(fp.get("technologies", []))
        print(f"{status}, {techs} tech(s) detected")

    # Stage 3: AI Analysis
    print(f"\n[Stage 3] Generating AI analysis prompt...")
    ai_prompt = generate_ai_analysis_prompt(
        args.target, subdomains, fingerprints
    )
    print(f"  ✅ Prompt generated ({len(ai_prompt)} chars)")
    print(f"\n{'─' * 60}")
    print("📋 Copy the prompt below into ChatGPT or Claude for AI analysis:")
    print(f"{'─' * 60}")
    print(ai_prompt)
    print(f"{'─' * 60}\n")

    # Stage 4: Report
    print(f"[Stage 4] Generating report...")
    report = generate_report(args.target, subdomains, fingerprints, ai_prompt)
    
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"  ✅ Report saved to: {args.output}")
    print(f"\n{'=' * 60}")
    print(f"📊 Summary: {len(subdomains)} subdomains, "
          f"{sum(1 for fp in fingerprints if fp.get('status'))} live hosts")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Suppress InsecureRequestWarning for lab use
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
