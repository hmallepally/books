import os
import re

chapters_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'
focus_dirs = ['08-algorithms-assessment', '09-q1-implementation', '10-q2-matrix-simulation', '11-q3-hashmaps-sliding-windows', '12-q4-optimization-dp']

replacements = {
    # Text Replacements
    r'\bQ1\b': 'Easy-tier',
    r'\bQ2\b': 'Medium-tier',
    r'\bQ3\b': 'Medium-Hard-tier',
    r'\bQ4\b': 'Hard-tier',
    'CodeSignal GCA': 'General Coding Assessments',
    'CodeSignal': 'automated testing platforms',
    'GCA': 'general coding assessment',
    
    # Specific code replacements
    'Map<Integer, Integer> prefCounts = new HashMap<>();': 'var prefCounts = new HashMap<Integer, Integer>();',
    'Deque<Integer> deque = new ArrayDeque<>();': 'var deque = new ArrayDeque<Integer>();',
    'int[] res = new int[nums.length - k + 1];': 'var res = new int[nums.length - k + 1];',
    'Deque<Character> stack = new ArrayDeque<>();': 'var stack = new ArrayDeque<Character>();',
    'int[] ans = new int[temps.length];': 'var ans = new int[temps.length];',
    'Queue<int[]> queue = new LinkedList<>();': 'var queue = new ArrayDeque<int[]>();',
    'Queue<Integer> queue = new LinkedList<>();': 'var queue = new ArrayDeque<Integer>();',
    'int[] inDegree = new int[numCourses];': 'var inDegree = new int[numCourses];',
    'List<List<Integer>> adj = new ArrayList<>();': 'var adj = new ArrayList<List<Integer>>();',
    'PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);': 'var pq = new PriorityQueue<int[]>((a, b) -> a[1] - b[1]);',
    'Map<Integer, Integer> dist = new HashMap<>();': 'var dist = new HashMap<Integer, Integer>();',
    'PriorityQueue<Integer> minHeap = new PriorityQueue<>();': 'var minHeap = new PriorityQueue<Integer>();',
    
    # new LinkedList<>() -> new ArrayDeque<>() for queue implementations
    'new LinkedList<>()': 'new ArrayDeque<>()'
}

def fix_code_and_text(text):
    for k, v in replacements.items():
        if k.startswith(r'\b'):
            text = re.sub(k, v, text)
        else:
            text = text.replace(k, v)
    
    # Safe generic var replacements (only inside methods, simple local vars)
    # We match: spaces + Type<Args> name = new Type<>();
    # Avoid fields by checking for standard indentation (4 or 8 spaces usually for locals).
    # Actually, the safest way is not to use generic regex because Java code in markdown is hard to distinguish accurately.
    # Let's just remove System.out.println.
    text = re.sub(r'^\s*System\.out\.println.*?;?\n', '', text, flags=re.MULTILINE)
    
    # Replace old testing platforms string
    text = text.replace('platforms such as automated testing platforms, HackerRank, LeetCode, Codility, or employer-proprietary assessments', 'platforms such as standardized online testing platforms, or employer-proprietary assessments')
    text = text.replace('automated testing platforms, some HackerRank', 'Standardized monotonic assessments')
    return text

for d in focus_dirs:
    filepath = os.path.join(chapters_dir, d, 'base.md')
    if not os.path.exists(filepath): continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = fix_code_and_text(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

print("Safely replaced.")
