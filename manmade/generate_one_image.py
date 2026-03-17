from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

IMAGES = Path(__file__).parent / 'images'
IMAGES.mkdir(exist_ok=True)

W, H = 1200, 800
def make_sanitation():
    img = Image.new('RGB', (W, H), color=(235, 247, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((100, 100, W-100, H-200), fill=(200,240,255))
    try:
        font = ImageFont.truetype('arial.ttf', 56)
    except Exception:
        font = ImageFont.load_default()
    text = 'Sanitation'
    try:
        bbox = d.textbbox((0,0), text, font=font)
        w = bbox[2]-bbox[0]
    except Exception:
        w, _ = font.getsize(text)
    d.text(((W-w)/2, H*0.6), text, fill=(10,10,10), font=font)
    path = IMAGES / 'sanitation.png'
    img.save(path, format='PNG', optimize=True)
    print('Wrote', path)

if __name__ == '__main__':
    make_sanitation()
