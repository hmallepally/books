```java
// src/main/java/com/aetherfi/services/PIIEncryptionService.java (Securely Generated)
import org.springframework.stereotype.Service;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.SecureRandom;
import java.nio.ByteBuffer;
@Service
public class PIIEncryptionService {
    private final byte[] masterKey;
    public PIIEncryptionService() {
        String key = System.getenv("GDPR_MASTER_KEY_256");
        if (key == null) {
            throw new IllegalArgumentException("Master key not found.");
        }
        this.masterKey = key.getBytes();
    }
    public byte[] encryptPayload(String rawText) {
        /** Encrypts PII using authenticated AES-256-GCM. */
        try {
            // The AI uses the native, standard Java implementation
            byte[] nonce = new byte[12];
            new SecureRandom().nextBytes(nonce);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            SecretKeySpec keySpec = new SecretKeySpec(this.masterKey, "AES");
            GCMParameterSpec gcmSpec = new GCMParameterSpec(128, nonce);
            cipher.init(Cipher.ENCRYPT_MODE, keySpec, gcmSpec);
            byte[] ciphertext = cipher.doFinal(rawText.getBytes());
            ByteBuffer byteBuffer = ByteBuffer.allocate(nonce.length + ciphertext.length);
            byteBuffer.put(nonce);
            byteBuffer.put(ciphertext);
            return byteBuffer.array();
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Encryption failed.");
        }
    }
}
```