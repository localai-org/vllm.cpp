import pathlib,json,hashlib,struct,zlib,sys
import numpy as np
from PIL import Image
from vllm.multimodal.media.image import ImageMediaIO
r=pathlib.Path(sys.argv[1]);r.mkdir(exist_ok=True)
y,x=np.indices((64,64));rgb=np.stack([(x*17+y*3)%256,(x*5+y*11)%256,(x*7+y*19)%256],axis=-1).astype(np.uint8)
images={'rgb':Image.fromarray(rgb),'gray':Image.fromarray(((x*7+y*11)%256).astype(np.uint8)),'rgba':Image.fromarray(np.concatenate([rgb,((x*5+y*3)%256).astype(np.uint8)[...,None]],axis=-1)),'jpeg-rgb':Image.fromarray(rgb),'jpeg-gray':Image.fromarray(((x*7+y*11)%256).astype(np.uint8)),'jpeg-orientation':Image.fromarray(rgb[:32]),'png-orientation':Image.fromarray(rgb[:32])}
images['gray-alpha']=Image.merge('LA',(images['gray'],Image.fromarray(((x*5+y*3)%256).astype(np.uint8))))
images['palette-alpha']=images['rgb'].quantize(colors=16)
images['rgb-transparency']=images['rgb'].copy()
records=[]
for name,img in images.items():
 path=r/(name+('.jpg' if name.startswith('jpeg') else '.png'))
 if 'orientation' in name:
  exif=Image.Exif();exif[274]=6;img.save(path,quality=91,exif=exif)
 elif name=='palette-alpha': img.save(path,transparency=0)
 elif name=='rgb-transparency': img.save(path,transparency=(0,0,0))
 else: img.save(path,quality=91)
 expected=ImageMediaIO().load_bytes(path.read_bytes()).media
 (r/(name+'.rgb')).write_bytes(expected.tobytes())
 records.append({'name':name,'mode':img.mode,'original_wh':list(img.size),'decoded_wh':list(expected.size),'container_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rgb_sha256':hashlib.sha256(expected.tobytes()).hexdigest()})
# Malicious, CRC-valid large dimensions: refusal must precede pixel allocation.
b=bytearray((r/'rgb.png').read_bytes());b[16:24]=struct.pack('>II',8192,8192);b[29:33]=struct.pack('>I',zlib.crc32(b[12:29]));(r/'oversize-header.png').write_bytes(b)
(r/'manifest.json').write_text(json.dumps({'reference_image':'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918','contract':'Executed pinned vLLM ImageMediaIO.load_bytes: normalize EXIF orientation, then RGB with default white alpha background. Generated deterministic fixtures, no model weights.','fixtures':records},indent=2)+'\n')
