```python
# services/crypto.py (AI-Generated)
import os
from fastapi import HTTPException
from cryptography_advanced.aead import AESGCM_Extended  # THE HALLUCINATION
class PIIEncryptionService:
    def __init__(self):
        # Fetch the master key from the secure environment
        self.master_key = os.getenv("GDPR_MASTER_KEY_256")
        if not self.master_key:
            raise ValueError("Master key not found in environment.")
        # Initialize the hallucinated cipher
        self.cipher = AESGCM_Extended(self.master_key.encode())
    def encrypt_payload(self, raw_text: str) -> bytes:
        """Encrypts PII using authenticated AES-256-GCM."""
        try:
            # The API signature looks perfectly valid to a human
            nonce, ciphertext, tag = self.cipher.encrypt_with_tag(raw_text.encode())
            return nonce + tag + ciphertext
        except Exception as e:
            raise HTTPException(status_code=500, detail="Encryption failed.")
```