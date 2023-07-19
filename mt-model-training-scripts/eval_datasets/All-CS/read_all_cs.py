import json

with open('All-CS.json','r') as f:
    data = json.load(f)

for line in data[:15]:
    print(line)
    print()