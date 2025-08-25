# Week 01: Array, String & Hash Table

## 📖 핵심 개념

### 1. Array & String 기초
```python
# 배열과 문자열 기본 연산
# Python에서 리스트는 동적 배열로 구현됨

# 인덱싱과 슬라이싱
arr = [1, 2, 3, 4, 5]
print(arr[0])      # 첫 번째 원소: 1
print(arr[-1])     # 마지막 원소: 5
print(arr[1:4])    # 슬라이싱 [2, 3, 4]
print(arr[::-1])   # 역순 [5, 4, 3, 2, 1]

# 문자열은 불변(immutable) 객체
s = "hello"
# s[0] = 'H'  # 에러! 문자열은 수정 불가
s = 'H' + s[1:]  # 새로운 문자열 생성으로 해결

# 리스트 컴프리헨션
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
```

### 2. Hash Table (딕셔너리/셋)
```python
# Python의 dict는 해시 테이블로 구현됨
# 평균 시간복잡도: 삽입, 삭제, 검색 모두 O(1)

# 기본 딕셔너리 사용
freq = {}
for num in [1, 2, 2, 3, 3, 3]:
    freq[num] = freq.get(num, 0) + 1
# freq = {1: 1, 2: 2, 3: 3}

# defaultdict 사용
from collections import defaultdict
freq = defaultdict(int)
for num in [1, 2, 2, 3, 3, 3]:
    freq[num] += 1

# Counter 사용
from collections import Counter
nums = [1, 2, 2, 3, 3, 3]
freq = Counter(nums)  # Counter({3: 3, 2: 2, 1: 1})
print(freq.most_common(2))  # [(3, 3), (2, 2)]

# set 활용
seen = set()
seen.add(1)
print(1 in seen)  # True - O(1) 검색
```

### 3. Two Pointer 기법
```python
# Two Pointer: 배열의 두 지점을 가리키는 포인터를 이용
# 정렬된 배열에서 특정 조건을 만족하는 쌍 찾기

def two_sum_sorted(nums, target):
    """정렬된 배열에서 합이 target인 두 수 찾기"""
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # 합이 작으면 왼쪽 포인터 이동
        else:
            right -= 1  # 합이 크면 오른쪽 포인터 이동
    
    return []

# 팰린드롬 확인
def is_palindrome(s):
    """문자열이 팰린드롬인지 확인"""
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

## 💡 주요 패턴

### 패턴 1: Sliding Window
- **사용 상황**: 연속된 부분 배열/문자열에서 특정 조건 만족하는 구간 찾기
- **시간복잡도**: O(n)
- **공간복잡도**: O(1) 또는 O(k) (k는 윈도우 크기)

```python
# 슬라이딩 윈도우 템플릿
def sliding_window_template(s):
    """고정 크기 k의 윈도우에서 최대 합 구하기"""
    k = 3  # 윈도우 크기
    n = len(s)
    if n < k:
        return 0
    
    # 초기 윈도우 설정
    window_sum = sum(s[:k])
    max_sum = window_sum
    
    # 윈도우 슬라이딩
    for i in range(k, n):
        window_sum = window_sum - s[i-k] + s[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# 가변 크기 윈도우 - 조건을 만족하는 최대/최소 길이
def variable_window(s, target):
    """합이 target 이상인 최소 길이 부분 배열"""
    left = 0
    current_sum = 0
    min_length = float('inf')
    
    for right in range(len(s)):
        current_sum += s[right]
        
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= s[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0
```

### 패턴 2: Hash Table을 활용한 빠른 검색
- **사용 상황**: 값의 존재 여부, 빈도수 계산, 중복 검사
- **시간복잡도**: O(n)
- **공간복잡도**: O(n)

```python
# Two Sum 패턴 - Hash Table 활용
def two_sum(nums, target):
    """배열에서 합이 target인 두 수의 인덱스 찾기"""
    seen = {}  # 값: 인덱스
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

# Anagram 그룹핑
def group_anagrams(strs):
    """anagram끼리 그룹핑"""
    from collections import defaultdict
    groups = defaultdict(list)
    
    for s in strs:
        # 정렬된 문자를 키로 사용
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())
```

## 🔑 Python 필수 문법

### 자료구조 관련
```python
# 리스트 기본 연산
arr = [1, 2, 3]
arr.append(4)        # 끝에 추가 O(1)
arr.pop()           # 끝에서 제거 O(1)
arr.insert(0, 0)    # 특정 위치 삽입 O(n)
arr.remove(2)       # 값으로 제거 O(n)

# 문자열 처리
s = "hello world"
s.split()           # ['hello', 'world']
'-'.join(['a','b']) # 'a-b'
s.replace('o', '0') # 'hell0 w0rld'
s.strip()           # 양쪽 공백 제거

# 딕셔너리 순회
d = {'a': 1, 'b': 2}
for key in d:              # 키 순회
    print(key, d[key])
for key, value in d.items():  # 키-값 순회
    print(key, value)
```

### 유용한 메서드/함수
```python
# collections 모듈
from collections import deque, defaultdict, Counter

# deque - 양방향 큐
dq = deque([1, 2, 3])
dq.appendleft(0)    # O(1)로 앞에 추가
dq.popleft()        # O(1)로 앞에서 제거

# defaultdict - 기본값이 있는 딕셔너리
dd = defaultdict(list)
dd['key'].append(1)  # 키가 없어도 자동으로 빈 리스트 생성

# Counter - 빈도수 계산
counter = Counter(['a', 'b', 'b', 'c', 'c', 'c'])
print(counter.most_common(1))  # [('c', 3)]

# 정렬
nums = [3, 1, 4, 1, 5]
sorted_nums = sorted(nums)  # 새 리스트 반환
nums.sort()                 # 원본 수정

# 커스텀 정렬
students = [('Alice', 85), ('Bob', 75), ('Charlie', 95)]
students.sort(key=lambda x: x[1])  # 점수로 정렬
```

## 🎯 LeetCode 추천 문제

### 필수 문제
- [ ] [1] Two Sum - Hash Table 기본
- [ ] [242] Valid Anagram - 문자 빈도수 계산
- [ ] [3] Longest Substring Without Repeating Characters - Sliding Window
- [ ] [11] Container With Most Water - Two Pointer
- [ ] [49] Group Anagrams - Hash 그룹핑

### 도전 문제
- [ ] [15] 3Sum - Two Pointer 응용
- [ ] [76] Minimum Window Substring - 고급 Sliding Window
- [ ] [438] Find All Anagrams in a String - Sliding Window + Hash

### 추가 연습
- [ ] [167] Two Sum II - Input Array Is Sorted
- [ ] [125] Valid Palindrome
- [ ] [387] First Unique Character in a String
- [ ] [383] Ransom Note
