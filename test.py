from collections import deque

a = deque([])
b = []
for i in range(10):
    a.append(i)
    b.append(i)

print(a[2] == b[2])
