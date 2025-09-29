#!/usr/bin/env python3
"""
Generate SVG diagrams for Six Sigma DMAIC and Kaizen processes
"""

import os

def create_six_sigma_dmaic_svg():
    """Create Six Sigma DMAIC process SVG diagram"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .phase-box { fill: #e1f5fe; stroke: #01579b; stroke-width: 2; }
      .activity-box { fill: #f3e5f5; stroke: #4a148c; stroke-width: 1; }
      .text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
      .title-text { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; text-anchor: middle; }
      .arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
    </marker>
  </defs>
  
  <!-- Title -->
  <text x="600" y="30" class="title-text" font-size="18">Six Sigma DMAIC Process Flow</text>
  
  <!-- Main Process Flow -->
  <rect x="50" y="80" width="120" height="60" rx="10" class="phase-box" />
  <text x="110" y="105" class="title-text">Define</text>
  <text x="110" y="120" class="text">Phase</text>
  
  <rect x="250" y="80" width="120" height="60" rx="10" class="phase-box" />
  <text x="310" y="105" class="title-text">Measure</text>
  <text x="310" y="120" class="text">Phase</text>
  
  <rect x="450" y="80" width="120" height="60" rx="10" class="phase-box" />
  <text x="510" y="105" class="title-text">Analyze</text>
  <text x="510" y="120" class="text">Phase</text>
  
  <rect x="650" y="80" width="120" height="60" rx="10" class="phase-box" />
  <text x="710" y="105" class="title-text">Improve</text>
  <text x="710" y="120" class="text">Phase</text>
  
  <rect x="850" y="80" width="120" height="60" rx="10" class="phase-box" />
  <text x="910" y="105" class="title-text">Control</text>
  <text x="910" y="120" class="text">Phase</text>
  
  <!-- Arrows between phases -->
  <line x1="170" y1="110" x2="250" y2="110" class="arrow" />
  <line x1="370" y1="110" x2="450" y2="110" class="arrow" />
  <line x1="570" y1="110" x2="650" y2="110" class="arrow" />
  <line x1="770" y1="110" x2="850" y2="110" class="arrow" />
  
  <!-- Continuous Improvement Loop -->
  <path d="M 970 110 Q 1000 110 1000 200 Q 1000 290 970 290" class="arrow" />
  <path d="M 50 290 Q 20 290 20 200 Q 20 110 50 110" class="arrow" />
  
  <!-- Activity Details -->
  <rect x="30" y="200" width="160" height="40" rx="5" class="activity-box" />
  <text x="110" y="220" class="text">Problem Identification</text>
  <text x="110" y="235" class="text">Stakeholder Analysis</text>
  
  <rect x="230" y="200" width="160" height="40" rx="5" class="activity-box" />
  <text x="310" y="220" class="text">Data Collection</text>
  <text x="310" y="235" class="text">Process Mapping</text>
  
  <rect x="430" y="200" width="160" height="40" rx="5" class="activity-box" />
  <text x="510" y="220" class="text">Root Cause Analysis</text>
  <text x="510" y="235" class="text">Statistical Analysis</text>
  
  <rect x="630" y="200" width="160" height="40" rx="5" class="activity-box" />
  <text x="710" y="220" class="text">Solution Design</text>
  <text x="710" y="235" class="text">Pilot Testing</text>
  
  <rect x="830" y="200" width="160" height="40" rx="5" class="activity-box" />
  <text x="910" y="220" class="text">Monitoring Systems</text>
  <text x="910" y="235" class="text">Documentation</text>
  
  <!-- Legend -->
  <rect x="50" y="300" width="200" height="100" rx="5" fill="#f5f5f5" stroke="#ccc" />
  <text x="150" y="320" class="title-text">Legend</text>
  <rect x="70" y="340" width="20" height="15" class="phase-box" />
  <text x="100" y="352" class="text">Main Process Phases</text>
  <rect x="70" y="360" width="20" height="15" class="activity-box" />
  <text x="100" y="372" class="text">Key Activities</text>
  <line x1="70" y1="385" x2="90" y2="385" class="arrow" />
  <text x="100" y="390" class="text">Process Flow</text>
</svg>'''
    
    return svg_content

def create_kaizen_cycle_svg():
    """Create Kaizen continuous improvement cycle SVG diagram"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .cycle-box { fill: #e1f5fe; stroke: #01579b; stroke-width: 2; }
      .activity-box { fill: #f3e5f5; stroke: #4a148c; stroke-width: 1; }
      .text { font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }
      .title-text { font-family: Arial, sans-serif; font-size: 13px; font-weight: bold; text-anchor: middle; }
      .arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
    </marker>
  </defs>
  
  <!-- Title -->
  <text x="600" y="30" class="title-text" font-size="18">Kaizen Continuous Improvement Cycle</text>
  
  <!-- Main Cycle -->
  <rect x="50" y="80" width="100" height="50" rx="10" class="cycle-box" />
  <text x="100" y="100" class="title-text">Identify</text>
  <text x="100" y="115" class="text">Opportunity</text>
  
  <rect x="200" y="80" width="100" height="50" rx="10" class="cycle-box" />
  <text x="250" y="100" class="title-text">Form Team</text>
  
  <rect x="350" y="80" width="100" height="50" rx="10" class="cycle-box" />
  <text x="400" y="100" class="title-text">Analyze</text>
  <text x="400" y="115" class="text">Current State</text>
  
  <rect x="500" y="80" width="100" height="50" rx="10" class="cycle-box" />
  <text x="550" y="100" class="title-text">Develop</text>
  <text x="550" y="115" class="text">Plan</text>
  
  <rect x="650" y="80" width="100" height="50" rx="10" class="cycle-box" />
  <text x="700" y="100" class="title-text">Implement</text>
  <text x="700" y="115" class="text">Changes</text>
  
  <rect x="800" y="80" width="100" height="50" rx="10" class="cycle-box" />
  <text x="850" y="100" class="title-text">Measure</text>
  <text x="850" y="115" class="text">Results</text>
  
  <rect x="950" y="80" width="100" height="50" rx="10" class="cycle-box" />
  <text x="1000" y="100" class="title-text">Standardize</text>
  
  <rect x="950" y="180" width="100" height="50" rx="10" class="cycle-box" />
  <text x="1000" y="200" class="title-text">Share</text>
  <text x="1000" y="215" class="text">Knowledge</text>
  
  <rect x="800" y="180" width="100" height="50" rx="10" class="cycle-box" />
  <text x="850" y="200" class="title-text">Plan Next</text>
  <text x="850" y="215" class="text">Improvement</text>
  
  <!-- Arrows -->
  <line x1="150" y1="105" x2="200" y2="105" class="arrow" />
  <line x1="300" y1="105" x2="350" y2="105" class="arrow" />
  <line x1="450" y1="105" x2="500" y2="105" class="arrow" />
  <line x1="600" y1="105" x2="650" y2="105" class="arrow" />
  <line x1="750" y1="105" x2="800" y2="105" class="arrow" />
  <line x1="900" y1="105" x2="950" y2="105" class="arrow" />
  <line x1="1000" y1="130" x2="1000" y2="180" class="arrow" />
  <line x1="950" y1="205" x2="800" y2="205" class="arrow" />
  <line x1="800" y1="180" x2="100" y2="180" class="arrow" />
  <line x1="100" y1="180" x2="100" y2="130" class="arrow" />
  
  <!-- Activity Details -->
  <rect x="30" y="280" width="140" height="60" rx="5" class="activity-box" />
  <text x="100" y="300" class="text">Employee Suggestions</text>
  <text x="100" y="315" class="text">Customer Feedback</text>
  <text x="100" y="330" class="text">Process Analysis</text>
  
  <rect x="190" y="280" width="140" height="60" rx="5" class="activity-box" />
  <text x="260" y="300" class="text">Cross-functional Team</text>
  <text x="260" y="315" class="text">Subject Matter Experts</text>
  <text x="260" y="330" class="text">Frontline Workers</text>
  
  <rect x="350" y="280" width="140" height="60" rx="5" class="activity-box" />
  <text x="420" y="300" class="text">Value Stream Mapping</text>
  <text x="420" y="315" class="text">Waste Identification</text>
  <text x="420" y="330" class="text">Root Cause Analysis</text>
  
  <rect x="510" y="280" width="140" height="60" rx="5" class="activity-box" />
  <text x="580" y="300" class="text">Solution Design</text>
  <text x="580" y="315" class="text">Resource Planning</text>
  <text x="580" y="330" class="text">Timeline Development</text>
  
  <rect x="670" y="280" width="140" height="60" rx="5" class="activity-box" />
  <text x="740" y="300" class="text">Pilot Testing</text>
  <text x="740" y="315" class="text">Full Implementation</text>
  <text x="740" y="330" class="text">Training</text>
  
  <rect x="830" y="280" width="140" height="60" rx="5" class="activity-box" />
  <text x="900" y="300" class="text">Performance Metrics</text>
  <text x="900" y="315" class="text">Quality Measures</text>
  <text x="900" y="330" class="text">Efficiency Gains</text>
  
  <!-- Legend -->
  <rect x="50" y="400" width="200" height="100" rx="5" fill="#f5f5f5" stroke="#ccc" />
  <text x="150" y="420" class="title-text">Legend</text>
  <rect x="70" y="440" width="20" height="15" class="cycle-box" />
  <text x="100" y="452" class="text">Main Process Steps</text>
  <rect x="70" y="460" width="20" height="15" class="activity-box" />
  <text x="100" y="472" class="text">Supporting Activities</text>
  <line x1="70" y1="485" x2="90" y2="485" class="arrow" />
  <text x="100" y="490" class="text">Process Flow</text>
</svg>'''
    
    return svg_content

def main():
    """Generate SVG diagrams"""
    print("🎨 Generating SVG Diagrams")
    print("=" * 40)
    
    # Ensure images directory exists
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"📁 Created images directory: {images_dir}")
    
    # Generate Six Sigma DMAIC diagram
    six_sigma_svg = create_six_sigma_dmaic_svg()
    six_sigma_file = f"{images_dir}/six_sigma_dmaic_process.svg"
    
    with open(six_sigma_file, 'w', encoding='utf-8') as f:
        f.write(six_sigma_svg)
    print(f"✅ Generated: {six_sigma_file}")
    
    # Generate Kaizen cycle diagram
    kaizen_svg = create_kaizen_cycle_svg()
    kaizen_file = f"{images_dir}/kaizen_improvement_cycle.svg"
    
    with open(kaizen_file, 'w', encoding='utf-8') as f:
        f.write(kaizen_svg)
    print(f"✅ Generated: {kaizen_file}")
    
    print("\n🎉 All SVG diagrams generated successfully!")
    print("\n📋 Next steps:")
    print("1. Update HTML to use the generated SVG images")
    print("2. Test the images in the browser")

if __name__ == "__main__":
    main()
