import re
import os

files_to_fix = [
    r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\08-algorithms-assessment\base.md',
    r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\09-q1-implementation\base.md',
    r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\10-q2-matrix-simulation\base.md',
    r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\11-q3-hashmaps-sliding-windows\base.md',
    r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\12-q4-optimization-dp\base.md'
]

def fix_code_and_text(text):
    # Pass 3: Text Replacements
    # Avoid replacing Q1 if it's part of a word? Q1/Q2/Q3/Q4 are usually separate.
    text = re.sub(r'\bQ1\b', 'Easy-tier', text)
    text = re.sub(r'\bQ2\b', 'Medium-tier', text)
    text = re.sub(r'\bQ3\b', 'Medium-Hard-tier', text)
    text = re.sub(r'\bQ4\b', 'Hard-tier', text)
    
    text = text.replace('CodeSignal GCA', 'General Coding Assessments')
    text = text.replace('CodeSignal', 'automated testing platforms')
    text = text.replace('GCA', 'general coding assessment')
    text = text.replace('platforms such as automated testing platforms, HackerRank, LeetCode, Codility, or employer-proprietary assessments', 'platforms such as standardized online testing platforms, or employer-proprietary assessments')

    # Pass 1: Java Replacements
    # var, Deque
    text = text.replace('Map<Integer, Integer> prefCounts = new HashMap<>();', 'var prefCounts = new HashMap<Integer, Integer>();')
    text = text.replace('Deque<Integer> deque = new ArrayDeque<>();', 'var deque = new ArrayDeque<Integer>();')
    text = text.replace('int[] res = new int[nums.length - k + 1];', 'var res = new int[nums.length - k + 1];')
    text = text.replace('Deque<Character> stack = new ArrayDeque<>();', 'var stack = new ArrayDeque<Character>();')
    text = text.replace('int[] ans = new int[temps.length];', 'var ans = new int[temps.length];')
    text = text.replace('Queue<int[]> queue = new LinkedList<>();', 'var queue = new ArrayDeque<int[]>();')
    text = text.replace('Queue<Integer> queue = new LinkedList<>();', 'var queue = new ArrayDeque<Integer>();')
    text = text.replace('int[] inDegree = new int[numCourses];', 'var inDegree = new int[numCourses];')
    text = text.replace('List<List<Integer>> adj = new ArrayList<>();', 'var adj = new ArrayList<List<Integer>>();')
    text = text.replace('PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);', 'var pq = new PriorityQueue<int[]>((a, b) -> a[1] - b[1]);')
    text = text.replace('Map<Integer, Integer> dist = new HashMap<>();', 'var dist = new HashMap<Integer, Integer>();')
    text = text.replace('PriorityQueue<Integer> minHeap = new PriorityQueue<>();', 'var minHeap = new PriorityQueue<Integer>();')
    
    # Generic var replacement for new objects
    text = re.sub(r'([A-Za-z0-9_<>]+)\s+([A-Za-z0-9_]+)\s*=\s*new\s+(HashMap|HashSet|ArrayList|LinkedList|ArrayDeque|PriorityQueue|int\[\]|boolean\[\]|char\[\])<.*?>\s*\(.*?\)\s*;', 
                  lambda m: f"var {m.group(2)} = new {m.group(3)}<>();", text)
                  
    # LinkedList to ArrayDeque for Queue/Deque
    text = text.replace('new LinkedList<>()', 'new ArrayDeque<>()')
    
    # Remove System.out.println
    text = re.sub(r'^\s*System\.out\.println.*?;?\n', '', text, flags=re.MULTILINE)
    
    # Also fix chapter cross refs if needed, though they might already be handled by Q1->Easy-tier.
    
    return text

for filepath in files_to_fix:
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    new_text = fix_code_and_text(text)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_text)

print("Done")
