#!/usr/bin/env python3
"""
Create the three missing images using PIL (Pillow) - Fixed version
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_ai_enhanced_kpi_system():
    """Create the AI-Enhanced KPI System diagram"""
    width, height = 1200, 900
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        text_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Colors
    center_color = (44, 62, 80)      # Dark blue
    hub_color = (52, 152, 219)       # Medium blue
    text_color = (44, 62, 80)        # Dark text
    
    # Title
    draw.text((width//2, 50), "AI-Enhanced KPI System", 
              font=title_font, fill=text_color, anchor="mm")
    
    # Center circle
    center_x, center_y = width//2, height//2
    draw.ellipse([center_x-80, center_y-80, center_x+80, center_y+80], 
                 fill=center_color, outline='white', width=3)
    draw.text((center_x, center_y), "AI-Enhanced\nKPI System\n\nIntelligent\nQuality\nManagement", 
              font=text_font, fill='white', anchor="mm")
    
    # Define component positions (6 components around center)
    components = [
        (center_x, center_y-200, "Real-time Quality\nMonitor\n📊 Live Metrics\n• Process monitoring\n• Alert systems\n• Trend analysis"),
        (center_x+180, center_y-120, "Predictive Quality\nAnalytics\n🔮 Future Insights\n• Quality forecasting\n• Risk assessment\n• Preventive actions"),
        (center_x+180, center_y+120, "Quality Performance\nInsights\n💡 Strategic Intelligence\n• Root cause analysis\n• Benchmark comparisons\n• ROI calculations"),
        (center_x, center_y+200, "Stakeholder\nCommunication\n👥 Automated Reports\n• Custom dashboards\n• Mobile access\n• Achievement tracking"),
        (center_x-180, center_y+120, "Automated Quality\nControl\n⚙️ Smart Automation\n• Inspection systems\n• Quality gates\n• Self-optimization"),
        (center_x-180, center_y-120, "Continuous\nImprovement Loop\n🔄 Optimization Engine\n• Learning algorithms\n• Process refinement\n• Performance enhancement")
    ]
    
    # Draw components
    for x, y, text in components:
        # Draw rounded rectangle
        draw.rounded_rectangle([x-120, y-60, x+120, y+60], radius=10, 
                              fill=hub_color, outline='white', width=2)
        draw.text((x, y), text, font=small_font, fill='white', anchor="mm")
        
        # Draw connecting line from center
        draw.line([center_x, center_y, x, y], fill=text_color, width=3)
    
    # Save image
    img.save('images/ai_enhanced_kpi_system.png', 'PNG', dpi=(300, 300))
    print('✅ Generated ai_enhanced_kpi_system.png')

def create_human_ai_collaboration():
    """Create the Human-AI Collaboration diagram"""
    width, height = 1400, 1000
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        header_font = ImageFont.truetype("arial.ttf", 20)
        text_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Colors
    human_accent = (39, 174, 96)     # Green
    ai_accent = (52, 152, 219)       # Blue
    collab_accent = (243, 156, 18)   # Orange
    
    # Title
    draw.text((width//2, 50), "Human-AI Collaboration Framework", 
              font=title_font, fill=(44, 62, 80), anchor="mm")
    
    # Human side
    draw.rounded_rectangle([50, 150, 550, 850], radius=20, 
                          fill=(232, 245, 232), outline=human_accent, width=3)
    draw.text((300, 200), "🧠 Human Expertise", font=header_font, 
              fill=human_accent, anchor="mm")
    
    # Human components
    human_components = [
        (300, 300, "Strategic Vision\n🎯 Long-term planning\n• Context understanding\n• Goal setting\n• Vision alignment"),
        (300, 450, "Creative Problem-Solving\n💡 Innovation & Creativity\n• Complex reasoning\n• Out-of-box thinking\n• Intuitive insights"),
        (300, 600, "Ethical Decision Making\n⚖️ Moral & Ethical Judgment\n• Value-based decisions\n• Ethical considerations\n• Social responsibility"),
        (300, 750, "Emotional Intelligence\n❤️ Human Connection\n• Empathy & understanding\n• Relationship building\n• Cultural awareness")
    ]
    
    for x, y, text in human_components:
        draw.rounded_rectangle([x-180, y-40, x+180, y+40], radius=10, 
                              fill='white', outline=human_accent, width=2)
        draw.text((x, y), text, font=small_font, fill=human_accent, anchor="mm")
    
    # AI side
    draw.rounded_rectangle([850, 150, 1350, 850], radius=20, 
                          fill=(232, 244, 253), outline=ai_accent, width=3)
    draw.text((1100, 200), "🤖 AI Capabilities", font=header_font, 
              fill=ai_accent, anchor="mm")
    
    # AI components
    ai_components = [
        (1100, 300, "Data Analysis\n📊 Pattern Recognition\n• Big data processing\n• Statistical analysis\n• Trend identification"),
        (1100, 450, "Predictive Analytics\n🔮 Future Forecasting\n• Scenario modeling\n• Risk assessment\n• Predictive insights"),
        (1100, 600, "Automated Processing\n⚙️ Operational Excellence\n• Process optimization\n• Real-time monitoring\n• Automated responses"),
        (1100, 750, "Machine Learning\n🧠 Continuous Learning\n• Model improvement\n• Adaptive algorithms\n• Performance optimization")
    ]
    
    for x, y, text in ai_components:
        draw.rounded_rectangle([x-180, y-40, x+180, y+40], radius=10, 
                              fill='white', outline=ai_accent, width=2)
        draw.text((x, y), text, font=small_font, fill=ai_accent, anchor="mm")
    
    # Collaboration hub
    draw.rounded_rectangle([620, 400, 780, 600], radius=20, 
                          fill=(255, 242, 204), outline=collab_accent, width=4)
    draw.text((700, 500), "🤝\nCollaboration\nHub\n\nSynergistic\nPartnership\n\nEnhanced\nCapabilities", 
              font=text_font, fill=collab_accent, anchor="mm")
    
    # Connecting arrows
    for i in range(4):
        y_pos = 300 + i * 150
        # Human to collaboration
        draw.line([530, y_pos, 620, 500], fill=collab_accent, width=3)
        # Collaboration to AI
        draw.line([780, 500, 850, y_pos], fill=collab_accent, width=3)
    
    img.save('images/human_ai_collaboration_diagram.png', 'PNG', dpi=(300, 300))
    print('✅ Generated human_ai_collaboration_diagram.png')

def create_ai_ethics_framework():
    """Create the AI Ethics Framework diagram"""
    width, height = 1200, 1200
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        header_font = ImageFont.truetype("arial.ttf", 20)
        text_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Colors
    center_color = (44, 62, 80)      # Dark blue
    fairness_color = (39, 174, 96)   # Green
    transparency_color = (52, 152, 219)  # Blue
    privacy_color = (155, 89, 182)   # Purple
    accountability_color = (243, 156, 18)  # Orange
    
    # Title
    draw.text((width//2, 50), "AI Ethics Framework", 
              font=title_font, fill=center_color, anchor="mm")
    
    # Center circle
    center_x, center_y = width//2, height//2
    draw.ellipse([center_x-100, center_y-100, center_x+100, center_y+100], 
                 fill=center_color, outline='white', width=4)
    draw.text((center_x, center_y), "AI Ethics\nFramework\n\n⚖️ Responsible AI\nImplementation\n\nBalanced &\nEthical AI\nSystems", 
              font=text_font, fill='white', anchor="mm")
    
    # Define quadrants
    quadrants = [
        (center_x-300, center_y-300, fairness_color, "🟢 Fairness\nEqual Treatment &\nNon-Discrimination", 
         ["Bias Detection\n• Algorithm auditing\n• Fairness metrics", "Equal Opportunities\n• Access equality\n• Inclusive design", "Transparent Decisions\n• Explainable outcomes\n• Clear criteria"]),
        (center_x+100, center_y-300, transparency_color, "🔵 Transparency\nOpenness &\nExplainability",
         ["Algorithm Transparency\n• Clear methodology\n• Process visibility", "Data Usage Disclosure\n• Data sources\n• Processing methods", "Decision Explainability\n• Reasoning clarity\n• Outcome justification"]),
        (center_x-300, center_y+100, privacy_color, "🟣 Privacy\nData Protection &\nConfidentiality",
         ["Data Minimization\n• Minimal data collection\n• Purpose limitation", "Security Measures\n• Encryption\n• Access controls", "User Consent\n• Informed consent\n• Opt-in/out options"]),
        (center_x+100, center_y+100, accountability_color, "🟠 Accountability\nResponsibility &\nOversight",
         ["Clear Responsibility\n• Defined roles\n• Decision ownership", "Monitoring & Auditing\n• Performance tracking\n• Regular audits", "Redress Mechanisms\n• Grievance procedures\n• Appeal processes"])
    ]
    
    # Draw quadrants
    for x, y, color, title, components in quadrants:
        # Background
        draw.rounded_rectangle([x-200, y-200, x+200, y+200], radius=20, 
                              fill=color, outline=color, width=3)
        
        # Title
        draw.text((x, y-150), title, font=header_font, fill=color, anchor="mm")
        
        # Components
        for i, component in enumerate(components):
            comp_y = y - 50 + i * 60
            draw.rounded_rectangle([x-150, comp_y-25, x+150, comp_y+25], radius=10, 
                                  fill='white', outline=color, width=2)
            draw.text((x, comp_y), component, font=small_font, fill=color, anchor="mm")
        
        # Connecting line to center
        draw.line([x, y, center_x, center_y], fill=color, width=2)
    
    img.save('images/ai_ethics_framework.png', 'PNG', dpi=(300, 300))
    print('✅ Generated ai_ethics_framework.png')

def main():
    """Generate all three missing images"""
    print("🎨 Creating missing images for the book...")
    print("=" * 50)
    
    try:
        create_ai_enhanced_kpi_system()
        create_human_ai_collaboration()
        create_ai_ethics_framework()
        
        print("=" * 50)
        print("🎉 All images generated successfully!")
        print("\n📁 Generated files:")
        print("  - images/ai_enhanced_kpi_system.png")
        print("  - images/human_ai_collaboration_diagram.png")
        print("  - images/ai_ethics_framework.png")
        print("\n✅ Ready to generate PDF!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()


