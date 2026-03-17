# Lab 8: Home Lab Setup Guide

**Book Reference**: Chapter 8 — *Building Your Arsenal: Career Path*

---

## Purpose

This guide walks you through setting up a complete ethical hacking lab environment on your own machine. No code to run — just step-by-step instructions to build the practice environment where you'll apply everything from the book.

## What You'll Build

```
Your Home Lab
├── Kali Linux VM          ← Your hacking OS (attacker machine)
├── Vulnerable Targets     ← Intentionally broken apps to practice on
│   ├── DVWA               ← Web app vulnerabilities
│   ├── Juice Shop         ← OWASP Top 10 playground
│   ├── Metasploitable 3   ← Network-level attacks
│   └── VulnHub machines   ← Boot-to-root challenges
├── Defense Tools          ← Blue team practice
│   ├── Wazuh (SIEM)       ← Open-source security monitoring
│   └── TheHive            ← Incident response platform
└── AI Tools               ← Force multiply everything
    ├── Python + Jupyter    ← For labs 3, 4, 6
    └── LLM Access          ← ChatGPT/Claude for analysis
```

---

## Step 1: Virtualization Platform

You need a hypervisor to run virtual machines safely isolated from your host:

| Platform | Cost | Best For |
|----------|------|----------|
| **VirtualBox** | Free | Beginners — easiest setup |
| **VMware Workstation Pro** | Free (personal) | Performance — industry standard |
| **Hyper-V** | Built into Windows Pro | Windows users — already installed |

**Download VirtualBox**: https://www.virtualbox.org/wiki/Downloads

### Recommended Host Specs
- **RAM**: 16 GB minimum (32 GB recommended)
- **Storage**: 100 GB free SSD space
- **CPU**: 4+ cores with virtualization enabled (VT-x/AMD-V)

---

## Step 2: Kali Linux (Your Hacker OS)

Kali Linux comes pre-loaded with 600+ security tools.

```bash
# Download from: https://www.kali.org/get-kali/
# Choose: "Kali Linux Virtual Machine" → VirtualBox/VMware

# Default credentials:
#   Username: kali
#   Password: kali
```

### Essential Tools Pre-installed
| Tool | Purpose | Book Chapter |
|------|---------|-------------|
| **Nmap** | Network scanning & discovery | Ch. 3, 5 |
| **Burp Suite CE** | Web vulnerability testing | Ch. 4, 5 |
| **Metasploit** | Exploitation framework | Ch. 5 |
| **Gobuster** | Directory brute-forcing | Ch. 3 |
| **Hydra** | Password brute-forcing | Ch. 5 |
| **Nikto** | Web server scanning | Ch. 4 |
| **Wireshark** | Packet analysis | Ch. 6 |

### First Things to Do
```bash
# Update Kali
sudo apt update && sudo apt full-upgrade -y

# Install Python extras for our labs
pip install requests beautifulsoup4 scikit-learn pandas numpy jupyter
```

---

## Step 3: Vulnerable Targets

### DVWA (Damn Vulnerable Web Application)
```bash
# Run with Docker (easiest)
docker run -d -p 80:80 vulnerables/web-dvwa

# Access: http://localhost
# Login: admin / password
# Set security level to "Low" to start
```

### OWASP Juice Shop
```bash
# Run with Docker
docker run -p 3000:3000 bkimminich/juice-shop

# Access: http://localhost:3000
# Built-in scoreboard tracks your progress
```

### Metasploitable 3
```bash
# Download from: https://github.com/rapid7/metasploitable3
# Requires Vagrant + VirtualBox

git clone https://github.com/rapid7/metasploitable3.git
cd metasploitable3
vagrant up
```

### TryHackMe & HackTheBox (Online)
- **TryHackMe** (https://tryhackme.com) — Beginner-friendly guided rooms
- **HackTheBox** (https://hackthebox.com) — Competitive CTF-style challenges

---

## Step 4: Blue Team Tools (Optional but Recommended)

### Wazuh (Open-Source SIEM)
```bash
# Quick start with Docker
git clone https://github.com/wazuh/wazuh-docker.git
cd wazuh-docker/single-node
docker-compose up -d

# Access: https://localhost:443
# Default: admin / SecretPassword
```

### Security Onion (Full SOC Suite)
- Download: https://securityonionsolutions.com/
- Includes: Suricata, Zeek, Kibana, TheHive, CyberChef

---

## Step 5: Network Isolation (Important!)

**Never connect vulnerable targets to the internet.** Use an isolated virtual network:

```
┌─────────────────────────────────────────┐
│           Host-Only Network             │
│        (e.g., 192.168.56.0/24)          │
│                                         │
│  ┌──────────┐     ┌──────────────────┐  │
│  │ Kali     │────▶│ Vulnerable       │  │
│  │ Attacker │     │ Targets          │  │
│  └──────────┘     └──────────────────┘  │
│                                         │
│  ┌──────────┐                           │
│  │ Wazuh    │  (monitors the targets)   │
│  │ SIEM     │                           │
│  └──────────┘                           │
└─────────────────────────────────────────┘
```

**VirtualBox Setup**:
1. Go to File → Host Network Manager → Create
2. Set adapter: 192.168.56.1, DHCP enabled
3. Assign all VMs to this "Host-Only" network

---

## Step 6: Your First Session

```
Week 1 Checklist:
  ✅ Install VirtualBox + Kali Linux
  ✅ Run DVWA in Docker
  ✅ Complete Lab 3 (AI Recon Pipeline)
  ✅ Complete first TryHackMe room
  
Week 2 Checklist:
  ✅ Run Juice Shop in Docker
  ✅ Complete Lab 4 (AI Vulnerability Scanner)
  ✅ Attempt Juice Shop challenges (at least 5)

Week 3 Checklist:
  ✅ Install Wazuh SIEM
  ✅ Complete Lab 6 (AI Alert Classifier)
  ✅ Set up Kali → DVWA → Wazuh monitoring pipeline

Week 4 Checklist:
  ✅ Sign up for HackTheBox
  ✅ Complete 1 "Easy" box
  ✅ Start studying for CompTIA Security+ or CEH
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| VM won't start | Enable VT-x/AMD-V in BIOS |
| Docker not found on Kali | `sudo apt install docker.io` |
| Can't reach targets from Kali | Check both VMs are on same Host-Only network |
| Kali is slow | Allocate at least 4 GB RAM and 2 CPU cores |
| DVWA shows blank page | Login and click "Create / Reset Database" |

---

## Key Takeaway

> *"You don't need expensive certifications to start. You need a laptop, virtualization software, and the discipline to practice consistently. The tools are free. The knowledge is open. The only barrier is showing up."*
>
> — Chapter 8, The Ethical Hacker's Playbook
