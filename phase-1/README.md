# Phase 1: 기초 자료구조와 핵심 알고리즘 (Week 1-3 압축)

## 📚 학습 목표
- [ ] Python 기본 자료구조 활용법 마스터
- [ ] Two Pointer, Sliding Window 패턴 이해
- [ ] Stack, Queue를 활용한 문제 해결
- [ ] 정렬 알고리즘과 Python sorted() 활용
- [ ] 코딩 테스트 필수 패턴 15개 습득

## 📖 Part 1: Array, String & Hash Table

### 1. Python 자료구조 핵심
```python
# 1. 리스트 (동적 배열)
arr = [1, 2, 3, 4, 5]
arr[0]         # O(1) - 인덱스 접근
arr[-1]        # O(1) - 마지막 원소
arr[1:4]       # O(k) - 슬라이싱
arr[::-1]      # O(n) - 역순
arr.append(6)  # O(1) amortized
arr.pop()      # O(1)
arr.insert(0, 0)  # O(n)

# 2. 문자열 (불변 객체)
s = "hello"
# s[0] = 'H'  # 에러! 
s = 'H' + s[1:]  # O(n) - 새 문자열 생성
''.join(['a', 'b', 'c'])  # O(n) - 효율적 결합

# 3. 딕셔너리/셋 (해시 테이블)
d = {}
d[key] = value  # O(1) average
key in d        # O(1) average
d.get(key, default)  # O(1) with default

# 4. Collections 모듈 필수
from collections import defaultdict, Counter, deque

# defaultdict - 자동 초기화
freq = defaultdict(int)
graph = defaultdict(list)

# Counter - 빈도수 계산
nums = [1, 2, 2, 3, 3, 3]
counter = Counter(nums)  # {3: 3, 2: 2, 1: 1}
counter.most_common(2)   # [(3, 3), (2, 2)]
```

### 2. Two Pointer 패턴
```python
def two_sum_sorted(nums: List[int], target: int) -> List[int]:
    """정렬된 배열에서 Two Sum - O(n)"""
    left, right = 0, len(nums) - 1
    
    while left < right:
        current = nums[left] + nums[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    return []

def three_sum(nums: List[int]) -> List[List[int]]:
    """3Sum 문제 - O(n²)"""
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:  # 중복 건너뛰기
            continue
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result
```

### 3. Sliding Window 패턴
```python
def longest_substring_without_repeat(s: str) -> int:
    """중복 없는 가장 긴 부분문자열 - O(n)"""
    char_index = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

def min_window_substring(s: str, t: str) -> str:
    """최소 윈도우 부분문자열 - O(n)"""
    if not s or not t:
        return ""
    
    need = Counter(t)
    have = defaultdict(int)
    
    left = 0
    formed = 0
    required = len(need)
    
    # (윈도우 길이, 왼쪽, 오른쪽)
    ans = float('inf'), None, None
    
    for right, char in enumerate(s):
        have[char] += 1
        
        if char in need and have[char] == need[char]:
            formed += 1
        
        while formed == required and left <= right:
            # 윈도우 업데이트
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            
            # 왼쪽 축소
            have[s[left]] -= 1
            if s[left] in need and have[s[left]] < need[s[left]]:
                formed -= 1
            left += 1
    
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
```

## 📖 Part 2: Linked List & Stack

### 4. Linked List 필수 패턴
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: ListNode) -> ListNode:
    """연결 리스트 뒤집기 - O(n)"""
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

def has_cycle(head: ListNode) -> bool:
    """Floyd's Cycle Detection - O(n)"""
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while fast and fast.next:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next
    
    return False

def find_middle(head: ListNode) -> ListNode:
    """중간 노드 찾기 - O(n)"""
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """두 정렬 리스트 병합 - O(n+m)"""
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next
```

### 5. Stack 패턴
```python
def is_valid_parentheses(s: str) -> bool:
    """괄호 유효성 검사 - O(n)"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack

def next_greater_element(nums: List[int]) -> List[int]:
    """Monotonic Stack - 다음 큰 원소 - O(n)"""
    n = len(nums)
    result = [-1] * n
    stack = []  # 인덱스 저장
    
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result

class MinStack:
    """최솟값 추적 스택 - 모든 연산 O(1)"""
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)
    
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]
```

## 📖 Part 3: Queue & Sorting

### 6. Queue와 BFS
```python
from collections import deque

def bfs_template(graph: Dict[int, List[int]], start: int) -> List[int]:
    """BFS 템플릿 - O(V + E)"""
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def level_order_traversal(root: TreeNode) -> List[List[int]]:
    """레벨별 순회 - O(n)"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### 7. Priority Queue (Heap)
```python
import heapq

def kth_largest(nums: List[int], k: int) -> int:
    """K번째 큰 원소 - O(n log k)"""
    # 최소 힙 사용 (크기 k 유지)
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]

def merge_k_sorted_lists(lists: List[ListNode]) -> ListNode:
    """K개 정렬 리스트 병합 - O(n log k)"""
    heap = []
    
    # 각 리스트의 첫 노드를 힙에 추가
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next

# 최대 힙 구현 (음수 사용)
def kth_smallest(nums: List[int], k: int) -> int:
    """K번째 작은 원소 - O(n log k)"""
    max_heap = []
    
    for num in nums:
        heapq.heappush(max_heap, -num)
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    
    return -max_heap[0]
```

### 8. 정렬 패턴
```python
# Python sorted() 활용 - Timsort O(n log n)

# 1. 기본 정렬
nums = [3, 1, 4, 1, 5]
sorted_nums = sorted(nums)  # 새 리스트
nums.sort()  # in-place

# 2. 커스텀 정렬
# 절댓값 기준
nums = [-4, -1, 0, 3, 10]
nums.sort(key=abs)

# 다중 기준
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
intervals.sort(key=lambda x: (x[0], x[1]))  # 시작점, 끝점 순

# 3. 구간 병합
def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """구간 병합 - O(n log n)"""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    
    return merged

# 4. Quick Select (K번째 원소) - 평균 O(n)
def quick_select(nums: List[int], k: int) -> int:
    """K번째 큰 원소 찾기 - Quick Select"""
    def partition(left: int, right: int, pivot_idx: int) -> int:
        pivot = nums[pivot_idx]
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        
        nums[right], nums[store_idx] = nums[store_idx], nums[right]
        return store_idx
    
    def select(left: int, right: int, k_smallest: int) -> int:
        if left == right:
            return nums[left]
        
        pivot_idx = (left + right) // 2
        pivot_idx = partition(left, right, pivot_idx)
        
        if k_smallest == pivot_idx:
            return nums[k_smallest]
        elif k_smallest < pivot_idx:
            return select(left, pivot_idx - 1, k_smallest)
        else:
            return select(pivot_idx + 1, right, k_smallest)
    
    return select(0, len(nums) - 1, len(nums) - k)
```

## 💡 코딩 테스트 필수 패턴 15선

### 배열/문자열 패턴
1. **Two Sum**: Hash Table O(n)
2. **Two/Three Pointer**: 정렬 후 포인터 이동
3. **Sliding Window**: 고정/가변 크기 윈도우
4. **Prefix Sum**: 구간 합 빠른 계산

### 해시 테이블 패턴
5. **Frequency Count**: Counter, defaultdict(int)
6. **Group by Pattern**: defaultdict(list)
7. **Set for Uniqueness**: 중복 제거, 존재 확인

### 연결 리스트 패턴
8. **Dummy Node**: 경계 케이스 처리 간소화
9. **Fast/Slow Pointer**: 중간 찾기, 순환 감지
10. **Reverse in Groups**: 부분 뒤집기

### 스택/큐 패턴
11. **Matching Pairs**: 괄호, 태그 매칭
12. **Monotonic Stack**: 다음 큰/작은 원소
13. **BFS Level Order**: 레벨별 처리

### 정렬/힙 패턴
14. **Custom Sort**: lambda, key 함수
15. **Top K Elements**: Heap 크기 제한

## 🔑 Python 코딩 테스트 필수 문법

### 입출력 최적화
```python
import sys
input = sys.stdin.readline  # 빠른 입력
sys.setrecursionlimit(10**6)  # 재귀 깊이 설정

# 여러 줄 입력
n = int(input())
arr = [list(map(int, input().split())) for _ in range(n)]
```

### 유용한 내장 함수
```python
# 최대/최소
max(arr), min(arr)
max(arr, key=lambda x: x[1])

# 정렬
sorted(arr, key=lambda x: (-x[0], x[1]))  # 다중 기준

# 이진 탐색
from bisect import bisect_left, bisect_right
idx = bisect_left(sorted_arr, target)

# 순열/조합
from itertools import permutations, combinations
list(permutations([1,2,3], 2))  # [(1,2), (1,3), (2,1), ...]
list(combinations([1,2,3], 2))   # [(1,2), (1,3), (2,3)]

# 누적합
from itertools import accumulate
list(accumulate([1,2,3,4]))  # [1, 3, 6, 10]
```

## 🎯 Phase 1 필수 문제 (20문제)

### Array & Hash Table (5문제)
- [1] Two Sum ⭐
- [3] Longest Substring Without Repeating Characters ⭐
- [11] Container With Most Water
- [15] 3Sum
- [49] Group Anagrams

### Linked List (4문제)
- [21] Merge Two Sorted Lists ⭐
- [141] Linked List Cycle ⭐
- [206] Reverse Linked List ⭐
- [2] Add Two Numbers

### Stack (4문제)
- [20] Valid Parentheses ⭐
- [155] Min Stack
- [739] Daily Temperatures
- [84] Largest Rectangle in Histogram

### Queue & BFS (3문제)
- [102] Binary Tree Level Order Traversal ⭐
- [200] Number of Islands
- [207] Course Schedule

### Sorting & Heap (4문제)
- [56] Merge Intervals ⭐
- [215] Kth Largest Element in an Array ⭐
- [23] Merge k Sorted Lists
- [347] Top K Frequent Elements

## 📝 핵심 체크리스트

### 자료구조 선택 가이드
- **빠른 검색**: Set, Dict → O(1)
- **순서 유지 + 양끝 작업**: deque → O(1)
- **최대/최소 추적**: Heap → O(log n)
- **정렬 유지**: bisect + list → O(log n) 검색

### 시간복잡도 목표
- n ≤ 10^3: O(n²) 가능
- n ≤ 10^5: O(n log n) 필요
- n ≤ 10^6: O(n) 필수
- n ≤ 10^9: O(log n) or O(1)

### 공간복잡도 최적화
- In-place 수정 가능하면 활용
- 슬라이싱은 복사 생성 주의
- Generator 활용 고려

### 디버깅 팁
1. 엣지 케이스: 빈 입력, 단일 원소
2. 경계값: 0, 음수, 최대값
3. 중복 처리 확인
4. Off-by-one 에러 주의