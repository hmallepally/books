f = open("c:/Users/hari/Documents/DBA/735/Week4/Walmart_Failures_Case_Study.md", "r", encoding="utf-8")
line1 = f.readline()
line19 = None
for i, line in enumerate(f, 2):
    if i == 19:
        line19 = line
        break
f.close()
print("Line 1:", repr(line1))
print("Line 19:", repr(line19))
