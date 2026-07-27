
from PIL import Image, ImageDraw
import os

def make_img(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 300), text, fill=(0,0,0))
    img.save(path)

make_img("chapters/02-industry-case-studies/visuals/domain_comparison.png", "Domain Comparison Infographic")
make_img("chapters/03-transition-roadmap/visuals/transition_roadmap.png", "Transition Roadmap Timeline")
make_img("chapters/08-data-analysis-sql/visuals/sql_joins.png", "SQL Joins Venn Diagrams")
make_img("chapters/10-stakeholder-management/visuals/stakeholder_grid.png", "Stakeholder Mapping Grid")
make_img("chapters/11-ai-copilot/visuals/ai_workflow.png", "AI Workflow Flowchart")
make_img("chapters/12-continuous-learning/visuals/flywheel.png", "Learning Flywheel Diagram")
make_img("chapters/13-tools-of-trade/visuals/tools_landscape.png", "Tool Landscape Infographic")

