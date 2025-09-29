#!/usr/bin/env python3
"""
Create simple 16x16 PNG placeholder images for emoji replacements
"""

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available, creating text-based placeholders instead")

import os

def create_placeholder_images():
    """Create simple placeholder images"""
    os.makedirs('images', exist_ok=True)
    
    if PIL_AVAILABLE:
        # Create actual PNG images
        images = {
            'rocket.png': create_rocket_png(),
            'lightbulb.png': create_lightbulb_png(),
            'tools.png': create_tools_png(),
            'book.png': create_book_png(),
            'chart.png': create_chart_png()
        }
        
        for filename, img in images.items():
            img.save(f'images/{filename}')
            print(f"Created images/{filename}")
    else:
        # Create text-based placeholders
        create_text_placeholders()

def create_rocket_png():
    """Create rocket PNG"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Simple rocket shape
    draw.rectangle([6, 2, 10, 12], fill='black')
    draw.polygon([(8, 0), (5, 3), (11, 3)], fill='black')
    draw.polygon([(6, 12), (4, 16), (6, 14)], fill='black')
    draw.polygon([(10, 12), (12, 16), (10, 14)], fill='black')
    
    return img

def create_lightbulb_png():
    """Create lightbulb PNG"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Simple lightbulb
    draw.ellipse([4, 2, 12, 10], fill='black')
    draw.rectangle([5, 10, 11, 12], fill='black')
    draw.rectangle([6, 12, 10, 14], fill='black')
    
    return img

def create_tools_png():
    """Create tools PNG"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Simple hammer and wrench
    draw.rectangle([2, 2, 6, 4], fill='black')
    draw.line([(4, 4), (4, 12)], fill='black', width=2)
    draw.line([(10, 2), (14, 2)], fill='black', width=2)
    draw.line([(14, 2), (14, 6)], fill='black', width=2)
    
    return img

def create_book_png():
    """Create book PNG"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Simple book
    draw.rectangle([3, 2, 13, 14], fill='black')
    draw.rectangle([4, 3, 12, 13], fill='white')
    draw.line([(5, 5), (11, 5)], fill='black', width=1)
    draw.line([(5, 7), (11, 7)], fill='black', width=1)
    draw.line([(5, 9), (11, 9)], fill='black', width=1)
    
    return img

def create_chart_png():
    """Create chart PNG"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Simple ascending chart
    draw.line([(2, 14), (14, 14)], fill='black', width=1)
    draw.line([(2, 2), (2, 14)], fill='black', width=1)
    draw.line([(3, 12), (6, 9), (9, 6), (12, 3)], fill='black', width=2)
    
    return img

def create_text_placeholders():
    """Create text-based placeholder files"""
    placeholders = {
        'rocket.png': 'ROCKET_ICON_16x16',
        'lightbulb.png': 'LIGHTBULB_ICON_16x16',
        'tools.png': 'TOOLS_ICON_16x16',
        'book.png': 'BOOK_ICON_16x16',
        'chart.png': 'CHART_ICON_16x16'
    }
    
    for filename, content in placeholders.items():
        with open(f'images/{filename}', 'w') as f:
            f.write(content)
        print(f"Created placeholder images/{filename}")

if __name__ == "__main__":
    create_placeholder_images()
    print("\nPlaceholder images created!")
    print("You can replace these with proper 16x16 PNG images later.")

