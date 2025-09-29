#!/usr/bin/env python3
"""
Create the three missing images for the book using matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

def create_ai_enhanced_kpi_system():
    """Create the AI-Enhanced KPI System diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define colors
    center_color = '#2c3e50'
    hub_color = '#3498db'
    data_color = '#ecf0f1'
    text_color = '#2c3e50'

    # Create center circle for AI-Enhanced KPI System
    center = Circle((5, 5), 0.8, facecolor=center_color, edgecolor='white', linewidth=2)
    ax.add_patch(center)
    ax.text(5, 5, 'AI-Enhanced\nKPI System\n\nIntelligent Quality\nManagement', 
            ha='center', va='center', fontsize=10, color='white', weight='bold')

    # Define the 6 main components
    components = [
        {'pos': (5, 7.5), 'text': 'Real-time Quality\nMonitor\n📊 Live Metrics\n• Process monitoring\n• Alert systems\n• Trend analysis', 'angle': 0},
        {'pos': (7.2, 6.8), 'text': 'Predictive Quality\nAnalytics\n🔮 Future Insights\n• Quality forecasting\n• Risk assessment\n• Preventive actions', 'angle': 60},
        {'pos': (7.8, 4.5), 'text': 'Quality Performance\nInsights\n💡 Strategic Intelligence\n• Root cause analysis\n• Benchmark comparisons\n• ROI calculations', 'angle': 120},
        {'pos': (7.2, 2.2), 'text': 'Stakeholder\nCommunication\n👥 Automated Reports\n• Custom dashboards\n• Mobile access\n• Achievement tracking', 'angle': 180},
        {'pos': (2.8, 2.2), 'text': 'Automated Quality\nControl\n⚙️ Smart Automation\n• Inspection systems\n• Quality gates\n• Self-optimization', 'angle': 240},
        {'pos': (2.2, 4.5), 'text': 'Continuous\nImprovement Loop\n🔄 Optimization Engine\n• Learning algorithms\n• Process refinement\n• Performance enhancement', 'angle': 300}
    ]

    # Create component boxes
    for comp in components:
        x, y = comp['pos']
        # Create rounded rectangle
        rect = FancyBboxPatch((x-1.2, y-0.8), 2.4, 1.6, 
                             boxstyle='round,pad=0.1', 
                             facecolor=hub_color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, comp['text'], ha='center', va='center', fontsize=8, color='white', weight='bold')

    # Add connecting lines from center to components
    for comp in components:
        x, y = comp['pos']
        ax.plot([5, x], [5, y], color='#34495e', linewidth=2, alpha=0.7)

    # Add data source boxes
    data_sources = [
        (6.5, 8.2, 'IoT Sensors\nQuality Systems\nCustomer Feedback'),
        (8.5, 6.5, 'Historical Data\nML Models\nExternal Factors'),
        (9, 4.5, 'Quality Databases\nIndustry Benchmarks\nFinancial Systems'),
        (8.5, 2.5, 'Management Systems\nCommunication Tools\nMobile Apps'),
        (2, 2.5, 'Quality Systems\nProduction Lines\nInspection Equipment'),
        (1, 4.5, 'Learning Systems\nOptimization Engines\nFeedback Loops')
    ]

    for x, y, text in data_sources:
        rect = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, 
                             boxstyle='round,pad=0.05', 
                             facecolor=data_color, edgecolor='#bdc3c7', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, color=text_color)

    # Add title
    ax.text(5, 9.5, 'AI-Enhanced KPI System', ha='center', va='center', 
            fontsize=16, weight='bold', color='#2c3e50')

    plt.tight_layout()
    plt.savefig('images/ai_enhanced_kpi_system.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print('✅ Generated ai_enhanced_kpi_system.png')

def create_human_ai_collaboration():
    """Create the Human-AI Collaboration diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define colors
    human_color = '#e8f5e8'
    ai_color = '#e8f4fd'
    collaboration_color = '#fff2cc'
    human_accent = '#27ae60'
    ai_accent = '#3498db'
    collab_accent = '#f39c12'

    # Human side
    ax.add_patch(FancyBboxPatch((0.5, 1), 4, 8, boxstyle='round,pad=0.2',
                               facecolor=human_color, edgecolor=human_accent, linewidth=2))
    ax.text(2.5, 9.2, '🧠 Human Expertise', ha='center', va='center', 
            fontsize=14, weight='bold', color=human_accent)

    # Human components
    human_components = [
        (2.5, 7.5, 'Strategic Vision\n🎯 Long-term planning\n• Context understanding\n• Goal setting\n• Vision alignment'),
        (2.5, 5.8, 'Creative Problem-Solving\n💡 Innovation & Creativity\n• Complex reasoning\n• Out-of-box thinking\n• Intuitive insights'),
        (2.5, 4.1, 'Ethical Decision Making\n⚖️ Moral & Ethical Judgment\n• Value-based decisions\n• Ethical considerations\n• Social responsibility'),
        (2.5, 2.4, 'Emotional Intelligence\n❤️ Human Connection\n• Empathy & understanding\n• Relationship building\n• Cultural awareness')
    ]

    for x, y, text in human_components:
        rect = FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2, boxstyle='round,pad=0.1',
                             facecolor='white', edgecolor=human_accent, linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color=human_accent)

    # AI side
    ax.add_patch(FancyBboxPatch((7.5, 1), 4, 8, boxstyle='round,pad=0.2',
                               facecolor=ai_color, edgecolor=ai_accent, linewidth=2))
    ax.text(9.5, 9.2, '🤖 AI Capabilities', ha='center', va='center', 
            fontsize=14, weight='bold', color=ai_accent)

    # AI components
    ai_components = [
        (9.5, 7.5, 'Data Analysis\n📊 Pattern Recognition\n• Big data processing\n• Statistical analysis\n• Trend identification'),
        (9.5, 5.8, 'Predictive Analytics\n🔮 Future Forecasting\n• Scenario modeling\n• Risk assessment\n• Predictive insights'),
        (9.5, 4.1, 'Automated Processing\n⚙️ Operational Excellence\n• Process optimization\n• Real-time monitoring\n• Automated responses'),
        (9.5, 2.4, 'Machine Learning\n🧠 Continuous Learning\n• Model improvement\n• Adaptive algorithms\n• Performance optimization')
    ]

    for x, y, text in ai_components:
        rect = FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2, boxstyle='round,pad=0.1',
                             facecolor='white', edgecolor=ai_accent, linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color=ai_accent)

    # Collaboration hub
    ax.add_patch(FancyBboxPatch((5.2, 3.5), 1.6, 3, boxstyle='round,pad=0.2',
                               facecolor=collaboration_color, edgecolor=collab_accent, linewidth=3))
    ax.text(6, 5, '🤝\nCollaboration\nHub\n\nSynergistic\nPartnership\n\nEnhanced\nCapabilities', 
            ha='center', va='center', fontsize=9, weight='bold', color=collab_accent)

    # Connecting arrows
    for i in range(4):
        y_human = 7.5 - i * 1.7
        y_ai = 7.5 - i * 1.7
        y_collab = 5
        
        # Human to collaboration
        ax.annotate('', xy=(5.2, y_collab), xytext=(4.3, y_human),
                   arrowprops=dict(arrowstyle='->', color=collab_accent, lw=2))
        # Collaboration to AI
        ax.annotate('', xy=(7.5, y_ai), xytext=(6.8, y_collab),
                   arrowprops=dict(arrowstyle='->', color=collab_accent, lw=2))

    plt.tight_layout()
    plt.savefig('images/human_ai_collaboration_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print('✅ Generated human_ai_collaboration_diagram.png')

def create_ai_ethics_framework():
    """Create the AI Ethics Framework diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define colors
    center_color = '#2c3e50'
    fairness_color = '#27ae60'
    transparency_color = '#3498db'
    privacy_color = '#9b59b6'
    accountability_color = '#f39c12'

    # Create center circle
    center = Circle((5, 5), 0.8, facecolor=center_color, edgecolor='white', linewidth=3)
    ax.add_patch(center)
    ax.text(5, 5, 'AI Ethics\nFramework\n\n⚖️ Responsible AI\nImplementation\n\nBalanced &\nEthical AI\nSystems', 
            ha='center', va='center', fontsize=9, color='white', weight='bold')

    # Define the four quadrants
    quadrants = [
        {'pos': (2.5, 7.5), 'color': fairness_color, 'title': '🟢 Fairness\nEqual Treatment &\nNon-Discrimination', 
         'components': ['Bias Detection\n• Algorithm auditing\n• Fairness metrics\n• Bias mitigation',
                       'Equal Opportunities\n• Access equality\n• Inclusive design\n• Diverse representation',
                       'Transparent Decisions\n• Explainable outcomes\n• Clear criteria\n• Auditable processes']},
        {'pos': (7.5, 7.5), 'color': transparency_color, 'title': '🔵 Transparency\nOpenness &\nExplainability',
         'components': ['Algorithm Transparency\n• Clear methodology\n• Process visibility\n• Documentation',
                       'Data Usage Disclosure\n• Data sources\n• Processing methods\n• Purpose clarity',
                       'Decision Explainability\n• Reasoning clarity\n• Outcome justification\n• User understanding']},
        {'pos': (2.5, 2.5), 'color': privacy_color, 'title': '🟣 Privacy\nData Protection &\nConfidentiality',
         'components': ['Data Minimization\n• Minimal data collection\n• Purpose limitation\n• Retention policies',
                       'Security Measures\n• Encryption\n• Access controls\n• Secure storage',
                       'User Consent\n• Informed consent\n• Opt-in/out options\n• Consent management']},
        {'pos': (7.5, 2.5), 'color': accountability_color, 'title': '🟠 Accountability\nResponsibility &\nOversight',
         'components': ['Clear Responsibility\n• Defined roles\n• Decision ownership\n• Responsibility chains',
                       'Monitoring & Auditing\n• Performance tracking\n• Regular audits\n• Compliance checks',
                       'Redress Mechanisms\n• Grievance procedures\n• Appeal processes\n• Remediation options']}
    ]

    # Create quadrants
    for quad in quadrants:
        x, y = quad['pos']
        color = quad['color']
        
        # Create quadrant background
        rect = FancyBboxPatch((x-2.2, y-2.2), 4.4, 4.4, boxstyle='round,pad=0.2',
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Add title
        ax.text(x, y+1.8, quad['title'], ha='center', va='center', 
                fontsize=10, weight='bold', color=color)
        
        # Add components
        for i, component in enumerate(quad['components']):
            comp_y = y + 0.5 - i * 0.8
            rect = FancyBboxPatch((x-1.8, comp_y-0.3), 3.6, 0.6, boxstyle='round,pad=0.05',
                                 facecolor='white', edgecolor=color, linewidth=1)
            ax.add_patch(rect)
            ax.text(x, comp_y, component, ha='center', va='center', 
                    fontsize=7, color='#2c3e50')

    # Add connecting lines between quadrants
    connections = [(2.5, 7.5, 7.5, 7.5), (7.5, 7.5, 7.5, 2.5), 
                   (7.5, 2.5, 2.5, 2.5), (2.5, 2.5, 2.5, 7.5)]
    
    for x1, y1, x2, y2 in connections:
        ax.plot([x1, x2], [y1, y2], color='#bdc3c7', linewidth=1, alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.savefig('images/ai_ethics_framework.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
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
        print("Please ensure matplotlib is installed: pip install matplotlib")

if __name__ == "__main__":
    main()


