```python
# Pipeline integration for OpenAI models
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": """You are a Zero-Trust Security Architect.
            
ABSOLUTE INVARIANTS (NEVER VIOLATE):
1. tenant_id isolation on ALL queries
2. Decimal for ALL financial math
3. bcrypt for ALL password hashing

BLAST RADIUS: You may ONLY modify files in src/transfer/
You are FORBIDDEN from touching src/auth/ or src/crypto/

STATE MACHINE: DRAFT -> PENDING -> SETTLED | FAILED
No other transitions are permitted."""
        },
        {
            "role": "user",
            "content": sdsd_prompt_with_code_context
        }
    ]
)
```