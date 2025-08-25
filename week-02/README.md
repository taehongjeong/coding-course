# Week 02: Linked List & Stack

## 📖 핵심 개념

### 1. Linked List (연결 리스트)
```python
# 노드 클래스 정의
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 단일 연결 리스트 기본 연산
class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        """끝에 노드 추가"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, val):
        """앞에 노드 추가"""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def delete_with_value(self, val):
        """값으로 노드 삭제"""
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        if current.next:
            current.next = current.next.next

# 리스트 순회
def traverse_list(head):
    """연결 리스트 순회"""
    current = head
    result = []
    while current:
        result.append(current.val)
        current = current.next
    return result
```

### 2. Stack (스택)
```python
# 리스트를 이용한 스택 구현
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """스택에 원소 추가 - O(1)"""
        self.items.append(item)
    
    def pop(self):
        """스택에서 원소 제거 - O(1)"""
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        """스택 최상단 원소 확인 - O(1)"""
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        """스택이 비어있는지 확인 - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """스택 크기 확인 - O(1)"""
        return len(self.items)

# collections.deque를 이용한 효율적인 스택
from collections import deque

stack = deque()
stack.append(1)    # push
stack.append(2)    # push
top = stack.pop()  # pop - 2 반환
```

### 3. Floyd's Cycle Detection (순환 감지)
```python
# Floyd's Tortoise and Hare Algorithm
def has_cycle(head):
    """연결 리스트의 순환 감지"""
    if not head or not head.next:
        return False
    
    slow = head       # 거북이 (한 칸씩)
    fast = head.next  # 토끼 (두 칸씩)
    
    while fast and fast.next:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next
    
    return False

# 순환 시작점 찾기
def detect_cycle_start(head):
    """순환이 시작되는 노드 찾기"""
    if not head or not head.next:
        return None
    
    # 1. 순환 감지
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # 순환 없음
    
    # 2. 시작점 찾기
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

## 💡 주요 패턴

### 패턴 1: Two Pointer in Linked List
- **사용 상황**: 리스트 중간 찾기, 순환 감지, 교집합 찾기
- **시간복잡도**: O(n)
- **공간복잡도**: O(1)

```python
# 연결 리스트 중간 노드 찾기
def find_middle(head):
    """slow/fast 포인터로 중간 노드 찾기"""
    if not head:
        return None
    
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# 끝에서 n번째 노드 찾기
def find_nth_from_end(head, n):
    """두 포인터 간격을 n으로 유지"""
    fast = slow = head
    
    # fast를 n칸 먼저 이동
    for _ in range(n):
        if not fast:
            return None
        fast = fast.next
    
    # 함께 이동
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

### 패턴 2: Monotonic Stack (단조 스택)
- **사용 상황**: 다음 큰 원소, 주식 가격, 히스토그램 넓이
- **시간복잡도**: O(n)
- **공간복잡도**: O(n)

```python
# 단조 감소 스택 - 다음 큰 원소 찾기
def next_greater_element(nums):
    """각 원소의 다음 큰 원소 찾기"""
    n = len(nums)
    result = [-1] * n
    stack = []  # 인덱스 저장
    
    for i in range(n):
        # 스택의 원소보다 현재 원소가 크면 pop
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result

# 괄호 매칭
def is_valid_parentheses(s):
    """괄호의 올바른 짝 확인"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # 닫는 괄호
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # 여는 괄호
            stack.append(char)
    
    return len(stack) == 0
```

## 🔑 Python 필수 문법

### 자료구조 관련
```python
# 연결 리스트 노드 생성
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 더미 노드 활용
dummy = ListNode(0)
dummy.next = head
# 작업 수행...
return dummy.next

# 리스트 역순 (in-place)
def reverse_list(head):
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev

# 스택 연산
stack = []
stack.append(item)    # push
top = stack.pop()     # pop
peek = stack[-1]      # peek (without removing)
is_empty = not stack  # empty check
```

### 유용한 메서드/함수
```python
# collections.deque - 양방향 큐/스택
from collections import deque

# deque as stack (더 효율적)
stack = deque()
stack.append(1)        # O(1) push
stack.pop()           # O(1) pop

# deque as queue
queue = deque()
queue.append(1)       # enqueue
queue.popleft()       # dequeue

# 문자열 처리
s = "({[]})"
for char in s:
    if char in "({[":
        # 여는 괄호 처리
        pass
    elif char in ")}]":
        # 닫는 괄호 처리
        pass

# 딕셔너리로 매핑
pairs = {'(': ')', '{': '}', '[': ']'}
```

## 🎯 LeetCode 추천 문제

### 필수 문제
- [ ] [20] Valid Parentheses - Stack 기본
- [ ] [21] Merge Two Sorted Lists - Linked List 병합
- [ ] [141] Linked List Cycle - Floyd's Algorithm
- [ ] [206] Reverse Linked List - 리스트 뒤집기
- [ ] [155] Min Stack - 최소값 추적 스택

### 도전 문제
- [ ] [2] Add Two Numbers - Linked List 연산
- [ ] [19] Remove Nth Node From End of List - Two Pointer
- [ ] [92] Reverse Linked List II - 부분 뒤집기

### 추가 연습
- [ ] [234] Palindrome Linked List
- [ ] [160] Intersection of Two Linked Lists
- [ ] [84] Largest Rectangle in Histogram
- [ ] [739] Daily Temperatures