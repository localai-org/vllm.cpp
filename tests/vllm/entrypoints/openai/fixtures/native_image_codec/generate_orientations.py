"""Generate asymmetric PNG/JPEG EXIF fixtures through pinned ImageMediaIO.

Run inside the reference environment. Separate output names preserve the
existing frozen codec captures. No learned model or GPU is loaded.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image
from vllm.multimodal.media.image import ImageMediaIO

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
y, x = np.indices((32, 64))
pixels = np.stack([(x * 17 + y * 3) % 256, (x * 5 + y * 11) % 256,
                   (x * 7 + y * 19) % 256], axis=-1).astype(np.uint8)
records = []
for byte_order, prefix in [('little', '<'), ('big', '>')]:
    for orientation in range(1, 9):
        # One bounded IFD0 SHORT orientation tag, exercising both TIFF parsers.
        tiff = (b'II' if byte_order == 'little' else b'MM') + struct.pack(prefix + 'HIH', 42, 8, 1)
        tiff += struct.pack(prefix + 'HHIHHI', 274, 3, 1, orientation, 0, 0)
        for extension, media_type in [('png', 'image/png'), ('jpg', 'image/jpeg')]:
            name = f'exif-{byte_order}-{orientation}-{extension}'
            path = root / (name + '.' + extension)
            Image.fromarray(pixels).save(path, quality=91, exif=b'Exif\x00\x00' + tiff)
            decoded = ImageMediaIO().load_bytes(path.read_bytes()).media
            expected_size = (32, 64) if orientation >= 5 else (64, 32)
            if decoded.size != expected_size or decoded.mode != 'RGB':
                raise RuntimeError('reference orientation or RGB contract changed')
            (root / (name + '.rgb')).write_bytes(decoded.tobytes())
            records.append({'name': name, 'media_type': media_type,
                            'orientation': orientation, 'tiff_byte_order': byte_order,
                            'original_wh': [64, 32], 'decoded_wh': list(decoded.size),
                            'container_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'rgb_sha256': hashlib.sha256(decoded.tobytes()).hexdigest()})
(root / 'orientations.json').write_text(json.dumps({
    'reference_image': 'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918',
    'contract': 'Executed pinned vLLM ImageMediaIO.load_bytes; all eight EXIF orientations, rectangular RGB input, both TIFF byte orders, PNG and JPEG.',
    'fixtures': records}, indent=2) + '\n')
print('captured', len(records), 'reference orientations')
