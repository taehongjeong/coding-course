# Week 03: Queue & Sorting

## 📖 핵심 개념

### 1. Queue (큐)
```python
# Queue: FIFO (First In First Out) 자료구조
# collections.deque를 사용한 효율적인 구현

from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """큐에 원소 추가 - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """큐에서 원소 제거 - O(1)"""
        if not self.is_empty():
            return self.items.popleft()
        return None
    
    def peek(self):
        """큐의 첫 번째 원소 확인 - O(1)"""
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self):
        """큐가 비어있는지 확인 - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """큐 크기 확인 - O(1)"""
        return len(self.items)

# deque를 직접 큐로 사용
queue = deque()
queue.append(1)      # enqueue
queue.append(2)      # enqueue
first = queue.popleft()  # dequeue - 1 반환

# Priority Queue (우선순위 큐)
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
    
    def push(self, item, priority):
        """우선순위와 함께 원소 추가 - O(log n)"""
        heapq.heappush(self.heap, (priority, item))
    
    def pop(self):
        """가장 높은 우선순위 원소 제거 - O(log n)"""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        return None
    
    def is_empty(self):
        return len(self.heap) == 0
```

### 2. Sorting Algorithms (정렬 알고리즘)
```python
# O(n²) 정렬 알고리즘 - 작은 데이터셋에 효과적

def bubble_sort(arr):
    """버블 정렬 - O(n²) 시간, O(1) 공간"""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break  # 이미 정렬된 경우 조기 종료
    return arr

def insertion_sort(arr):
    """삽입 정렬 - O(n²) 시간, O(1) 공간
    거의 정렬된 데이터에 효율적"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def selection_sort(arr):
    """선택 정렬 - O(n²) 시간, O(1) 공간"""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# O(n log n) 정렬 알고리즘 - 대용량 데이터에 효율적

def merge_sort(arr):
    """병합 정렬 - O(n log n) 시간, O(n) 공간
    안정 정렬, 일관된 성능"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """두 정렬된 배열 병합"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    """퀵 정렬 - 평균 O(n log n), 최악 O(n²)
    평균적으로 가장 빠른 정렬"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
```

### 3. Python sorted() & Custom Sorting
```python
# Python 내장 정렬 (Timsort - O(n log n))
# 매우 최적화되어 있어 실무에서 권장

# 기본 정렬
arr = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_arr = sorted(arr)  # 새 리스트 반환
arr.sort()  # 원본 리스트 정렬

# 역순 정렬
arr.sort(reverse=True)
sorted_desc = sorted(arr, reverse=True)

# 커스텀 정렬 - key 함수 사용
# 절댓값 기준 정렬
nums = [-4, -1, 0, 3, 10]
sorted_abs = sorted(nums, key=abs)

# 문자열 길이 기준 정렬
words = ["apple", "pie", "z", "banana"]
sorted_len = sorted(words, key=len)

# 다중 기준 정렬
students = [
    ("Alice", 85),
    ("Bob", 75),
    ("Charlie", 85),
    ("Dave", 75)
]
# 점수 내림차순, 이름 오름차순
sorted_students = sorted(students, key=lambda x: (-x[1], x[0]))

# 딕셔너리 정렬
scores = {"Alice": 85, "Bob": 75, "Charlie": 90}
# 값 기준 정렬
sorted_by_value = sorted(scores.items(), key=lambda x: x[1])
# 키 기준 정렬
sorted_by_key = sorted(scores.items())

# 클래스/객체 정렬
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __repr__(self):
        return f"Student({self.name}, {self.grade})"

students = [
    Student("Alice", 85),
    Student("Bob", 75),
    Student("Charlie", 90)
]

# grade 기준 정렬
students.sort(key=lambda s: s.grade)

# operator.attrgetter 사용
from operator import attrgetter
students.sort(key=attrgetter('grade'))

# functools.cmp_to_key 사용 (고급)
from functools import cmp_to_key

def compare(a, b):
    """커스텀 비교 함수"""
    if a.grade != b.grade:
        return b.grade - a.grade  # grade 내림차순
    return -1 if a.name < b.name else 1  # name 오름차순

students.sort(key=cmp_to_key(compare))
```

## 💡 주요 패턴

### 패턴 1: BFS (Breadth-First Search) with Queue
- **사용 상황**: 레벨 순회, 최단 경로, 그래프 탐색
- **시간복잡도**: O(V + E) - V: 정점, E: 간선
- **공간복잡도**: O(V)

```python
# BFS 템플릿 - 레벨별 처리
from collections import deque

def bfs_by_level(graph, start):
    """레벨별 BFS 탐색"""
    visited = set([start])
    queue = deque([start])
    level = 0
    
    while queue:
        # 현재 레벨의 모든 노드 처리
        level_size = len(queue)
        print(f"Level {level}: ", end="")
        
        for _ in range(level_size):
            node = queue.popleft()
            print(node, end=" ")
            
            # 인접 노드 탐색
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        print()  # 레벨 구분
        level += 1
    
    return visited

# 최단 경로 찾기 (무가중 그래프)
def shortest_path(graph, start, end):
    """BFS를 이용한 최단 경로"""
    if start == end:
        return 0
    
    visited = set([start])
    queue = deque([(start, 0)])  # (노드, 거리)
    
    while queue:
        node, dist = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return dist + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # 경로 없음
```

### 패턴 2: Merge Intervals (구간 병합)
- **사용 상황**: 겹치는 구간 병합, 스케줄링, 시간 범위 처리
- **시간복잡도**: O(n log n) - 정렬 때문
- **공간복잡도**: O(n)

```python
def merge_intervals(intervals):
    """겹치는 구간 병합"""
    if not intervals:
        return []
    
    # 시작점 기준 정렬
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # 겹치는 경우 병합
        if current[0] <= last[1]:
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # 겹치지 않으면 추가
            merged.append(current)
    
    return merged

# 회의실 배정 문제
def can_attend_meetings(intervals):
    """모든 회의 참석 가능 여부"""
    if not intervals:
        return True
    
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False  # 겹치는 회의 존재
    
    return True

# 필요한 회의실 개수
import heapq

def min_meeting_rooms(intervals):
    """최소 회의실 개수 계산"""
    if not intervals:
        return 0
    
    # 시작 시간 기준 정렬
    intervals.sort(key=lambda x: x[0])
    
    # 끝나는 시간을 저장하는 최소 힙
    rooms = []
    heapq.heappush(rooms, intervals[0][1])
    
    for interval in intervals[1:]:
        # 가장 빨리 끝나는 회의실이 비어있으면 재사용
        if rooms[0] <= interval[0]:
            heapq.heappop(rooms)
        
        # 새 회의 추가
        heapq.heappush(rooms, interval[1])
    
    return len(rooms)
```

## 🔑 Python 필수 문법

### 자료구조 관련
```python
# collections.deque - 양방향 큐
from collections import deque

# deque 생성 및 연산
dq = deque([1, 2, 3])
dq.append(4)        # 오른쪽 추가: [1, 2, 3, 4]
dq.appendleft(0)    # 왼쪽 추가: [0, 1, 2, 3, 4]
dq.pop()            # 오른쪽 제거: 4
dq.popleft()        # 왼쪽 제거: 0
dq.rotate(1)        # 오른쪽 회전: [3, 1, 2]
dq.rotate(-1)       # 왼쪽 회전: [1, 2, 3]

# heapq - 최소 힙 (우선순위 큐)
import heapq

# 힙 생성 및 연산
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
min_val = heapq.heappop(heap)  # 1

# 리스트를 힙으로 변환
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)  # O(n)

# 최대 힙 구현 (음수 사용)
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -5)
max_val = -heapq.heappop(max_heap)  # 5

# n개의 최소/최대 원소
nums = [3, 1, 4, 1, 5, 9, 2, 6]
smallest_3 = heapq.nsmallest(3, nums)  # [1, 1, 2]
largest_3 = heapq.nlargest(3, nums)    # [9, 6, 5]

# queue.PriorityQueue (thread-safe)
from queue import PriorityQueue

pq = PriorityQueue()
pq.put((2, "task2"))  # (우선순위, 값)
pq.put((1, "task1"))
pq.put((3, "task3"))
task = pq.get()  # (1, "task1")
```

### 유용한 메서드/함수
```python
# 정렬 관련
from operator import itemgetter, attrgetter
from functools import cmp_to_key

# itemgetter - 인덱스/키 기준 정렬
data = [(1, 'b'), (2, 'a'), (1, 'a')]
sorted_data = sorted(data, key=itemgetter(0, 1))

# attrgetter - 속성 기준 정렬
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

points = [Point(1, 2), Point(3, 1), Point(2, 3)]
sorted_points = sorted(points, key=attrgetter('x', 'y'))

# bisect - 이진 탐색 (정렬된 리스트)
from bisect import bisect_left, bisect_right, insort

sorted_list = [1, 3, 4, 4, 6, 8]
pos_left = bisect_left(sorted_list, 4)   # 2 (왼쪽 삽입 위치)
pos_right = bisect_right(sorted_list, 4)  # 4 (오른쪽 삽입 위치)
insort(sorted_list, 5)  # 정렬 유지하며 삽입

# 구간 관련
def overlaps(interval1, interval2):
    """두 구간이 겹치는지 확인"""
    return interval1[0] < interval2[1] and interval2[0] < interval1[1]

def merge_two_intervals(interval1, interval2):
    """두 구간 병합"""
    return [min(interval1[0], interval2[0]), 
            max(interval1[1], interval2[1])]
```

## 🎯 LeetCode 추천 문제

### 필수 문제
- [ ] [26] Remove Duplicates from Sorted Array - 정렬된 배열 처리
- [ ] [27] Remove Element - 배열 원소 제거
- [ ] [88] Merge Sorted Array - 정렬된 배열 병합
- [ ] [225] Implement Stack using Queues - 큐로 스택 구현
- [ ] [232] Implement Queue using Stacks - 스택으로 큐 구현

### 도전 문제
- [ ] [56] Merge Intervals - 구간 병합
- [ ] [147] Insertion Sort List - 연결 리스트 삽입 정렬
- [ ] [215] Kth Largest Element in an Array - K번째 큰 원소

### 추가 연습
- [ ] [75] Sort Colors - Dutch National Flag
- [ ] [179] Largest Number - 커스텀 정렬
- [ ] [252] Meeting Rooms - 회의실 배정
- [ ] [253] Meeting Rooms II - 최소 회의실 개수