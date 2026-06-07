```python
# services/crypto.py (Securely Generated)
import os
from fastapi import HTTPException
# The AI correctly imports the whitelisted package
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
class PIIEncryptionService:
    def __init__(self):
        self.master_key = os.getenv("GDPR_MASTER_KEY_256")
        if not self.master_key:
            raise ValueError("Master key not found.")
    def encrypt_payload(self, raw_text: str) -> bytes:
        """Encrypts PII using authenticated AES-256-GCM."""
        try:
            # The AI uses the slightly more verbose, but native, implementation
            nonce = os.urandom(12)
            cipher = Cipher(algorithms.AES(self.master_key.encode()), modes.GCM(nonce))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(raw_text.encode()) + encryptor.finalize()
            return nonce + encryptor.tag + ciphertext
        except Exception:
            raise HTTPException(status_code=500, detail="Encryption failed.")
```