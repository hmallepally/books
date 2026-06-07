```java
// src/main/java/com/aetherfi/services/PIIEncryptionService.java (AI-Generated)
import org.springframework.stereotype.Service;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.security.crypto.advanced.AESGCM_Extended; // THE HALLUCINATION
@Service
public class PIIEncryptionService {
    private final byte[] masterKey;
    private final AESGCM_Extended cipher;
    public PIIEncryptionService() {
        // Fetch the master key from the secure environment
        String key = System.getenv("GDPR_MASTER_KEY_256");
        if (key == null) {
            throw new IllegalArgumentException("Master key not found in environment.");
        }
        this.masterKey = key.getBytes();
        // Initialize the hallucinated cipher
        this.cipher = new AESGCM_Extended(this.masterKey);
    }
    public byte[] encryptPayload(String rawText) {
        /** Encrypts PII using authenticated AES-256-GCM. */
        try {
            // The API signature looks perfectly valid to a human
            byte[][] result = this.cipher.encryptWithTag(rawText.getBytes());
            byte[] nonce = result[0];
            byte[] ciphertext = result[1];
            byte[] tag = result[2];
            // Concatenate arrays
            byte[] combined = new byte[nonce.length + tag.length + ciphertext.length];
            System.arraycopy(nonce, 0, combined, 0, nonce.length);
            System.arraycopy(tag, 0, combined, nonce.length, tag.length);
            System.arraycopy(ciphertext, 0, combined, nonce.length + tag.length, ciphertext.length);
            return combined;
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Encryption failed.");
        }
    }
}
```