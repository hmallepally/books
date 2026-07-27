```java
package com.aurapay.security;

import java.nio.ByteBuffer;
import java.security.SecureRandom;
import java.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * Utility for AES-GCM 256-bit encryption/decryption of sensitive PII or PAN data,
 * adhering to PCI-DSS requirements.
 */
public class TokenizationUtility {

    private static final String ALGORITHM = "AES/GCM/NoPadding";
    private static final int TAG_LENGTH_BITS = 128;
    private static final int IV_LENGTH_BYTES = 12;
    private static final SecureRandom SECURE_RANDOM = new SecureRandom();

    /**
     * Encrypts the plaintext data using the provided 256-bit key.
     * Returns a URL-safe Base64-encoded string containing [IV][Ciphertext][Tag].
     */
    public static String encrypt(String plaintext, byte[] keyBytes) throws Exception {
        if (plaintext == null || keyBytes == null || keyBytes.length != 32) {
            throw new IllegalArgumentException("Invalid plaintext or key size. Key must be 256-bit.");
        }

        // 1. Generate a secure random Initialization Vector (IV)
        byte[] iv = new byte[IV_LENGTH_BYTES];
        SECURE_RANDOM.nextBytes(iv);

        // 2. Initialize Cipher in ENCRYPT_MODE
        SecretKey key = new SecretKeySpec(keyBytes, "AES");
        Cipher cipher = Cipher.getInstance(ALGORITHM);
        GCMParameterSpec spec = new GCMParameterSpec(TAG_LENGTH_BITS, iv);
        cipher.init(Cipher.ENCRYPT_MODE, key, spec);

        // 3. Encrypt the data
        byte[] ciphertext = cipher.doFinal(plaintext.getBytes("UTF-8"));

        // 4. Combine IV and Ciphertext into a single payload
        ByteBuffer byteBuffer = ByteBuffer.allocate(iv.length + ciphertext.length);
        byteBuffer.put(iv);
        byteBuffer.put(ciphertext);
        byte[] encryptedPayload = byteBuffer.array();

        // 5. Encode to Base64
        return Base64.getUrlEncoder().withoutPadding().encodeToString(encryptedPayload);
    }

    /**
     * Decrypts the Base64-encoded payload using the provided 256-bit key.
     */
    public static String decrypt(String base64Payload, byte[] keyBytes) throws Exception {
        if (base64Payload == null || keyBytes == null || keyBytes.length != 32) {
            throw new IllegalArgumentException("Invalid payload or key size. Key must be 256-bit.");
        }

        // 1. Decode from Base64
        byte[] encryptedPayload = Base64.getUrlDecoder().decode(base64Payload);

        // 2. Extract the IV
        if (encryptedPayload.length < IV_LENGTH_BYTES) {
            throw new IllegalArgumentException("Ciphertext payload is truncated or invalid.");
        }
        ByteBuffer byteBuffer = ByteBuffer.wrap(encryptedPayload);
        byte[] iv = new byte[IV_LENGTH_BYTES];
        byteBuffer.get(iv);

        // 3. Extract the actual ciphertext
        byte[] ciphertext = new byte[byteBuffer.remaining()];
        byteBuffer.get(ciphertext);

        // 4. Initialize Cipher in DECRYPT_MODE
        SecretKey key = new SecretKeySpec(keyBytes, "AES");
        Cipher cipher = Cipher.getInstance(ALGORITHM);
        GCMParameterSpec spec = new GCMParameterSpec(TAG_LENGTH_BITS, iv);
        cipher.init(Cipher.DECRYPT_MODE, key, spec);

        // 5. Decrypt and convert to String
        byte[] decryptedBytes = cipher.doFinal(ciphertext);
        return new String(decryptedBytes, "UTF-8");
    }
}
```
