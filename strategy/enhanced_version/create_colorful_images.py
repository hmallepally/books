#!/usr/bin/env python3
"""
Create colorful 16x16 PNG images to replace emojis in the Living Strategy book.
These images will be vibrant and colorful for better visual appeal.
"""

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Please install Pillow: pip install Pillow")

import os

def create_colorful_images():
    """Create colorful PNG images"""
    if not PIL_AVAILABLE:
        print("Cannot create images without PIL. Please install Pillow.")
        return
    
    os.makedirs('images', exist_ok=True)
    
    # Create each colorful image
    images = {
        'rocket.png': create_colorful_rocket(),
        'lightbulb.png': create_colorful_lightbulb(),
        'tools.png': create_colorful_tools(),
        'book.png': create_colorful_book(),
        'chart.png': create_colorful_chart()
    }
    
    # Save images
    for filename, img in images.items():
        img.save(f'images/{filename}')
        print(f"Created colorful images/{filename}")
    
    print("\nAll colorful images created successfully!")

def create_colorful_rocket():
    """Create a colorful rocket icon"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Rocket body (blue gradient effect)
    draw.rectangle([6, 2, 10, 12], fill=(30, 144, 255))  # Dodger blue
    
    # Rocket nose (red)
    draw.polygon([(8, 0), (5, 3), (11, 3)], fill=(220, 20, 60))  # Crimson
    
    # Rocket fins (dark blue)
    draw.polygon([(6, 12), (4, 16), (6, 14)], fill=(25, 25, 112))  # Midnight blue
    draw.polygon([(10, 12), (12, 16), (10, 14)], fill=(25, 25, 112))  # Midnight blue
    
    # Rocket flame (orange/yellow)
    draw.line([(7, 14), (7, 16)], fill=(255, 165, 0), width=2)  # Orange
    draw.line([(9, 14), (9, 16)], fill=(255, 215, 0), width=2)  # Gold
    
    # Add some highlights
    draw.line([(7, 3), (7, 11)], fill=(135, 206, 250), width=1)  # Light sky blue
    draw.line([(9, 3), (9, 11)], fill=(135, 206, 250), width=1)  # Light sky blue
    
    return img

def create_colorful_lightbulb():
    """Create a colorful lightbulb icon"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Lightbulb body (bright yellow)
    draw.ellipse([4, 2, 12, 10], fill=(255, 255, 0))  # Yellow
    
    # Lightbulb base (dark gray)
    draw.rectangle([5, 10, 11, 12], fill=(105, 105, 105))  # Dim gray
    
    # Lightbulb screw base (darker gray)
    draw.rectangle([6, 12, 10, 14], fill=(64, 64, 64))  # Dark gray
    
    # Light rays (bright yellow)
    draw.line([(2, 4), (4, 4)], fill=(255, 255, 0), width=2)  # Yellow
    draw.line([(12, 4), (14, 4)], fill=(255, 255, 0), width=2)  # Yellow
    draw.line([(2, 6), (4, 6)], fill=(255, 255, 0), width=2)  # Yellow
    draw.line([(12, 6), (14, 6)], fill=(255, 255, 0), width=2)  # Yellow
    
    # Add filament (orange)
    draw.line([(7, 5), (9, 5)], fill=(255, 140, 0), width=1)  # Dark orange
    draw.line([(7, 7), (9, 7)], fill=(255, 140, 0), width=1)  # Dark orange
    
    return img

def create_colorful_tools():
    """Create a colorful tools icon"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Hammer head (brown)
    draw.rectangle([2, 2, 6, 4], fill=(139, 69, 19))  # Saddle brown
    
    # Hammer handle (dark brown)
    draw.line([(4, 4), (4, 12)], fill=(101, 67, 33), width=3)  # Dark brown
    
    # Wrench (silver/gray)
    draw.line([(10, 2), (14, 2)], fill=(192, 192, 192), width=3)  # Silver
    draw.line([(14, 2), (14, 6)], fill=(192, 192, 192), width=3)  # Silver
    draw.line([(10, 6), (14, 6)], fill=(192, 192, 192), width=3)  # Silver
    
    # Add some highlights
    draw.line([(3, 3), (5, 3)], fill=(160, 82, 45), width=1)  # Sienna
    
    return img

def create_colorful_book():
    """Create a colorful book icon"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Book cover (green)
    draw.rectangle([3, 2, 13, 14], fill=(34, 139, 34))  # Forest green
    
    # Book pages (white)
    draw.rectangle([4, 3, 12, 13], fill=(255, 255, 255))  # White
    
    # Book spine (dark green)
    draw.line([(3, 2), (3, 14)], fill=(0, 100, 0), width=2)  # Dark green
    
    # Book text lines (dark green)
    draw.line([(5, 5), (11, 5)], fill=(0, 100, 0), width=1)  # Dark green
    draw.line([(5, 7), (11, 7)], fill=(0, 100, 0), width=1)  # Dark green
    draw.line([(5, 9), (11, 9)], fill=(0, 100, 0), width=1)  # Dark green
    draw.line([(5, 11), (11, 11)], fill=(0, 100, 0), width=1)  # Dark green
    
    # Add bookmark (red)
    draw.line([(13, 2), (13, 8)], fill=(220, 20, 60), width=2)  # Crimson
    
    return img

def create_colorful_chart():
    """Create a colorful chart icon"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Chart axes (dark gray)
    draw.line([(2, 14), (14, 14)], fill=(64, 64, 64), width=2)  # Dark gray
    draw.line([(2, 2), (2, 14)], fill=(64, 64, 64), width=2)  # Dark gray
    
    # Chart line (green ascending)
    draw.line([(3, 12), (6, 9), (9, 6), (12, 3)], fill=(0, 255, 0), width=3)  # Lime green
    
    # Data points (different colors)
    draw.ellipse([(2, 11), (4, 13)], fill=(255, 0, 0))  # Red
    draw.ellipse([(5, 8), (7, 10)], fill=(255, 165, 0))  # Orange
    draw.ellipse([(8, 5), (10, 7)], fill=(255, 255, 0))  # Yellow
    draw.ellipse([(11, 2), (13, 4)], fill=(0, 255, 0))  # Green
    
    # Add grid lines (light gray)
    draw.line([(5, 2), (5, 14)], fill=(200, 200, 200), width=1)  # Light gray
    draw.line([(8, 2), (8, 14)], fill=(200, 200, 200), width=1)  # Light gray
    draw.line([(11, 2), (11, 14)], fill=(200, 200, 200), width=1)  # Light gray
    
    return img

if __name__ == "__main__":
    create_colorful_images()

