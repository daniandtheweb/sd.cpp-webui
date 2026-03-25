"""sd.cpp-webui - Utilities for video processing"""

import struct
import os
from typing import Tuple, Optional


def get_avi_resolution(filepath: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Reads the width and height of an AVI file purely in native Python
    by searching the binary header for the 'avih' chunk.
    """
    if not os.path.exists(filepath):
        return None, None

    try:
        with open(filepath, 'rb') as f:
            # The 'avih' chunk is always in the header, safely within the first 2KB
            header_data = f.read(2048)

            # Verify it's actually an AVI file
            if not header_data.startswith(b'RIFF') or b'AVI ' not in header_data[:12]:
                return None, None

            # Find the 'avih' chunk signature
            avih_idx = header_data.find(b'avih')

            if avih_idx != -1:
                # The dwWidth property starts exactly 40 bytes after the 'a' in 'avih'
                # (4 bytes for 'avih', 4 bytes for chunk size, then 32 bytes of offset)
                width_start = avih_idx + 40

                if width_start + 8 <= len(header_data):
                    # Unpack two 32-bit little-endian integers
                    width = struct.unpack('<I', header_data[width_start: width_start+4])[0]
                    height = struct.unpack('<I', header_data[width_start+4: width_start+8])[0]
                    return width, height

    except Exception as e:
        print(f"Error parsing AVI header for {filepath}: {e}")

    return None, None
