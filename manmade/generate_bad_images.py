from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(__file__).parent
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

bad_entries = [
    "Asbestos",
    "Leaded Gasoline",
    "Chemical Weapons",
    "Thalidomide",
    "Agent Orange",
    "Single-use Plastics",
    "DDT",
    "Fossil Fuels",
    "Industrial Waste",
    "Nuclear Weapons",
]

W = 1200
H = 800

def make_bad_image(name, filename):
    img = Image.new('RGB', (W, H), color=(255, 245, 235))
    d = ImageDraw.Draw(img)

    # simple rounded rectangle and a warning icon
    d.rounded_rectangle((100, 80, W-100, H-180), radius=40, fill=(254, 202, 202), outline=(120,20,20))
    cx, cy = W//2, H//2 - 40
    # small triangle warning
    tri = [(cx, cy-90), (cx-70, cy+40), (cx+70, cy+40)]
    d.polygon(tri, fill=(234,88,12))

    try:
        font = ImageFont.truetype('arial.ttf', 44)
        small = ImageFont.truetype('arial.ttf', 20)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    try:
        bbox = d.textbbox((0,0), name, font=font)
        w = bbox[2] - bbox[0]
    except Exception:
        w, _ = font.getsize(name)
    d.text(((W-w)/2, H*0.6), name, fill=(40, 20, 20), font=font)

    caption = "Manmade — inventions"
    try:
        bbox2 = d.textbbox((0,0), caption, font=small)
        w2 = bbox2[2] - bbox2[0]
    except Exception:
        w2, _ = small.getsize(caption)
    d.text(((W-w2)/2, H*0.88), caption, fill=(80,80,80), font=small)

    path = IMAGES / filename
    img.save(path, format='PNG', optimize=True)
    print('Wrote', path)

if __name__ == '__main__':
    for e in bad_entries:
        fname = e.lower().replace(' ', '_').replace('-', '_') + '.png'
        make_bad_image(e, fname)
