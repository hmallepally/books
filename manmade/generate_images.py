from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(__file__).parent
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

entries = [
    "Agriculture",
    "Wheel",
    "Writing",
    "Printing Press",
    "Electricity",
    "Vaccination",
    "Steam Engine",
    "Telephone",
    "Computer",
    "Internet",
]

W = 1200
H = 800

def make_image(name, filename):
    img = Image.new('RGB', (W, H), color=(240, 248, 255))
    d = ImageDraw.Draw(img)

    # large circle as simple icon
    cx, cy = W//2, H//3
    r = 140
    d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(16, 185, 129), outline=(10,10,10))

    # title text
    try:
        font = ImageFont.truetype('arial.ttf', 56)
        small = ImageFont.truetype('arial.ttf', 28)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    try:
        bbox = d.textbbox((0,0), name, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = font.getsize(name)
    d.text(((W-w)/2, H*0.6), name, fill=(12, 12, 12), font=font)

    caption = "Manmade — inventions"
    try:
        bbox2 = d.textbbox((0,0), caption, font=small)
        w2 = bbox2[2] - bbox2[0]
    except Exception:
        w2, _ = small.getsize(caption)
    d.text(((W-w2)/2, H*0.85), caption, fill=(80,80,80), font=small)

    path = IMAGES / filename
    img.save(path, format='PNG', optimize=True)
    print('Wrote', path)

if __name__ == '__main__':
    for e in entries:
        fname = e.lower().replace(' ', '_') + '.png'
        make_image(e, fname)
