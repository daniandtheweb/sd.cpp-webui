"""sd.cpp-webui - Image encryption module"""

import os
import io
from PIL import Image


class ImageEncryption:
    """处理图片的加密和解密（兼容 sd-cli 的 XOR 加密）"""
    
    def __init__(self, password="123"):
        self.password = password
    
    def _generate_key(self):
        """生成 256 字节密钥（与 decrypt.js 相同算法）"""
        key = bytearray(256)
        password_bytes = self.password.encode('utf-8')
        for i in range(256):
            key[i] = password_bytes[i % len(password_bytes)] ^ (i & 0xFF)
        return bytes(key)
    
    def decrypt_image_file(self, encrypted_path):
        """解密图片文件并返回 PIL Image 对象"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        # XOR 解密
        key = self._generate_key()
        decrypted_data = bytearray(len(encrypted_data))
        for i in range(len(encrypted_data)):
            decrypted_data[i] = encrypted_data[i] ^ key[i % len(key)]
        
        # 转换为 PIL Image
        return Image.open(io.BytesIO(bytes(decrypted_data)))
