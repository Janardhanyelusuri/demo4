from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import base64


# AES encryption function
def encrypt_data(data, key):
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes long for AES-256 encryption")

    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()

    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    encrypted_data_base64 = base64.b64encode(encrypted_data).decode()

    return encrypted_data_base64


# AES decryption function
def decrypt_data(encrypted_data, key):
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes long for AES-256 encryption")

    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()

    encrypted_data_bytes = base64.b64decode(encrypted_data)

    decrypted_padded_data = decryptor.update(encrypted_data_bytes) + decryptor.finalize()

    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

    return data.decode()
