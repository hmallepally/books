```csharp
using System;
using System.Security.Cryptography;
using System.Text;

namespace AuraPay.Security
{
    /// <summary>
    /// Utility for AES-GCM 256-bit encryption/decryption of sensitive PII or PAN data,
    /// adhering to PCI-DSS requirements.
    /// </summary>
    public static class TokenizationUtility
    {
        private const int NonceSize = 12; // 96-bit nonce/IV
        private const int TagSize = 16;   // 128-bit authentication tag

        /// <summary>
        /// Encrypts the plaintext data using the provided 256-bit key.
        /// Returns a URL-safe Base64-encoded string containing [Nonce][Ciphertext][Tag].
        /// </summary>
        public static string Encrypt(string plaintext, byte[] keyBytes)
        {
            if (string.IsNullOrEmpty(plaintext) || keyBytes == null || keyBytes.Length != 32)
            {
                throw new ArgumentException("Invalid plaintext or key size. Key must be 256-bit.");
            }

            byte[] plaintextBytes = Encoding.UTF8.GetBytes(plaintext);
            byte[] nonce = new byte[NonceSize];
            RandomNumberGenerator.Fill(nonce);

            byte[] ciphertext = new byte[plaintextBytes.Length];
            byte[] tag = new byte[TagSize];

            using (var aesGcm = new AesGcm(keyBytes, TagSize))
            {
                aesGcm.Encrypt(nonce, plaintextBytes, ciphertext, tag);
            }

            // Combine Nonce + Ciphertext + Tag
            byte[] result = new byte[NonceSize + ciphertext.Length + TagSize];
            Buffer.BlockCopy(nonce, 0, result, 0, NonceSize);
            Buffer.BlockCopy(ciphertext, 0, result, NonceSize, ciphertext.Length);
            Buffer.BlockCopy(tag, 0, result, NonceSize + ciphertext.Length, TagSize);

            return Convert.ToBase64String(result).Replace('+', '-').Replace('/', '_').TrimEnd('=');
        }

        /// <summary>
        /// Decrypts the Base64-encoded payload using the provided 256-bit key.
        /// </summary>
        public static string Decrypt(string base64Payload, byte[] keyBytes)
        {
            if (string.IsNullOrEmpty(base64Payload) || keyBytes == null || keyBytes.Length != 32)
            {
                throw new ArgumentException("Invalid payload or key size. Key must be 256-bit.");
            }

            // Restore base64 padding
            string incoming = base64Payload.Replace('-', '+').Replace('_', '/');
            switch (incoming.Length % 4)
            {
                case 2: incoming += "=="; break;
                case 3: incoming += "="; break;
            }
            byte[] encryptedPayload = Convert.FromBase64String(incoming);

            if (encryptedPayload.Length < NonceSize + TagSize)
            {
                throw new ArgumentException("Ciphertext payload is truncated or invalid.");
            }

            byte[] nonce = new byte[NonceSize];
            byte[] tag = new byte[TagSize];
            int ciphertextLength = encryptedPayload.Length - NonceSize - TagSize;
            byte[] ciphertext = new byte[ciphertextLength];

            Buffer.BlockCopy(encryptedPayload, 0, nonce, 0, NonceSize);
            Buffer.BlockCopy(encryptedPayload, NonceSize, ciphertext, 0, ciphertextLength);
            Buffer.BlockCopy(encryptedPayload, NonceSize + ciphertextLength, tag, 0, TagSize);

            byte[] decryptedBytes = new byte[ciphertextLength];

            using (var aesGcm = new AesGcm(keyBytes, TagSize))
            {
                aesGcm.Decrypt(nonce, ciphertext, tag, decryptedBytes);
            }

            return Encoding.UTF8.GetString(decryptedBytes);
        }
    }
}
```
