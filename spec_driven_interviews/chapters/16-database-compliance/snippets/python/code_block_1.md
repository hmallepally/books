```python
import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class TokenizationUtility:
    """
    Utility for AES-GCM 256-bit encryption/decryption of sensitive PII or PAN data,
    adhering to PCI-DSS requirements.
    """
    
    @staticmethod
    def encrypt(plaintext: str, key_bytes: bytes) -> str:
        if not plaintext or len(key_bytes) != 32:
            raise ValueError("Invalid plaintext or key size. Key must be 256-bit.")
            
        # 1. Generate a secure random Initialization Vector (IV)
        iv = os.urandom(12)
        
        # 2. Encrypt using AES-GCM
        aesgcm = AESGCM(key_bytes)
        ciphertext = aesgcm.encrypt(iv, plaintext.encode('utf-8'), None)
        
        # 3. Combine IV and Ciphertext and base64-encode
        payload = iv + ciphertext
        return base64.urlsafe_b64encode(payload).decode('utf-8').rstrip('=')

    @staticmethod
    def decrypt(base64_payload: str, key_bytes: bytes) -> str:
        if not base64_payload or len(key_bytes) != 32:
            raise ValueError("Invalid payload or key size. Key must be 256-bit.")
            
        # 1. Pad and decode the base64 string
        missing_padding = len(base64_payload) % 4
        if missing_padding:
            base64_payload += '=' * (4 - missing_padding)
        encrypted_payload = base64.urlsafe_b64decode(base64_payload.encode('utf-8'))
        
        if len(encrypted_payload) < 12:
            raise ValueError("Ciphertext payload is truncated or invalid.")
            
        # 2. Extract IV and Ciphertext
        iv = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        
        # 3. Decrypt using AES-GCM
        aesgcm = AESGCM(key_bytes)
        decrypted_bytes = aesgcm.decrypt(iv, ciphertext, None)
        return decrypted_bytes.decode('utf-8')
```
