# Week 04: Recursion & Binary Tree 기초

## 📖 핵심 개념

### 1. 재귀(Recursion) 기초부터 심화까지
```python
# 재귀의 3가지 핵심 요소
# 1. Base Case (종료 조건)
# 2. Recursive Case (재귀 호출)
# 3. Progress toward Base Case (종료 조건에 가까워짐)

def factorial(n):
    """팩토리얼 - 재귀의 기본 예제
    시간복잡도: O(n)
    공간복잡도: O(n) - 콜스택
    """
    if n <= 1:  # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

# 재귀의 문제점: 중복 계산
def fibonacci_naive(n):
    """순진한 피보나치 구현
    시간복잡도: O(2^n) - 지수적 증가!
    공간복잡도: O(n)
    
    문제: fib(5) 계산 시
    - fib(3)은 2번 계산
    - fib(2)는 3번 계산
    - fib(1)은 5번 계산
    """
    if n <= 1:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)

# 해결책 1: 메모이제이션
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memo(n):
    """메모이제이션으로 최적화
    시간복잡도: O(n) - 각 값을 한 번만 계산
    공간복잡도: O(n) - 캐시 저장
    """
    if n <= 1:
        return n
    return fibonacci_memo(n-1) + fibonacci_memo(n-2)

# 해결책 2: Bottom-up DP (반복문)
def fibonacci_iterative(n):
    """반복문으로 구현
    시간복잡도: O(n)
    공간복잡도: O(1) - 상수 공간
    """
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr

# Python의 재귀 제한
import sys
# 기본 재귀 제한: 1000
# 필요시 늘릴 수 있지만 주의 필요
sys.setrecursionlimit(10**6)

# 실용적인 해결책: 반복문으로 변환
def factorial_iterative(n):
    """반복문으로 구현한 팩토리얼
    시간복잡도: O(n)
    공간복잡도: O(1) - 스택 사용 안 함
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 재귀 사용 시 주의사항
# 1. Python의 기본 재귀 제한은 1000
# 2. 깊은 재귀가 필요한 경우 sys.setrecursionlimit() 사용
# 3. 하지만 가능하면 반복문이나 메모이제이션 사용 권장
```

### 2. Divide and Conquer (분할 정복)
```python
# 분할 정복의 3단계
# 1. Divide: 문제를 작은 부분으로 나눔
# 2. Conquer: 재귀적으로 부분 문제 해결
# 3. Combine: 부분 해를 합쳐서 전체 해 구성

# 예제 1: 병합 정렬 (Merge Sort)
def merge_sort(arr):
    """병합 정렬 - 분할 정복의 전형적 예
    시간복잡도: O(n log n)
    공간복잡도: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Conquer
    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)
    
    # Combine
    return merge(left_sorted, right_sorted)

def merge(left, right):
    """두 정렬된 배열을 병합"""
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

# 예제 2: 빠른 거듭제곱 (Binary Exponentiation)
def power(x, n):
    """분할 정복으로 거듭제곱 계산
    시간복잡도: O(log n) - 일반 O(n)보다 훨씬 빠름!
    
    아이디어: x^n = (x^(n/2))^2 if n is even
              x^n = x * x^(n-1) if n is odd
    """
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)
    
    # 재귀 버전
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    else:
        return x * power(x, n - 1)

def power_iterative(x, n):
    """반복문 버전 - 더 효율적
    비트 연산 활용: x^13 = x^8 * x^4 * x^1 (13 = 1101₂)
    """
    if n == 0:
        return 1
    if n < 0:
        x, n = 1/x, -n
    
    result = 1
    while n > 0:
        if n & 1:  # n이 홀수면
            result *= x
        x *= x
        n >>= 1  # n을 2로 나눔
    return result

# 예제 3: 최대 부분 배열 (Maximum Subarray)
def max_subarray_divide_conquer(arr):
    """분할 정복으로 최대 부분 배열 합 찾기
    시간복잡도: O(n log n)
    """
    def max_crossing_sum(arr, left, mid, right):
        """중간을 지나는 최대 합"""
        # 왼쪽 부분의 최대 합
        left_sum = float('-inf')
        sum_val = 0
        for i in range(mid, left - 1, -1):
            sum_val += arr[i]
            left_sum = max(left_sum, sum_val)
        
        # 오른쪽 부분의 최대 합
        right_sum = float('-inf')
        sum_val = 0
        for i in range(mid + 1, right + 1):
            sum_val += arr[i]
            right_sum = max(right_sum, sum_val)
        
        return left_sum + right_sum
    
    def helper(arr, left, right):
        if left == right:
            return arr[left]
        
        mid = (left + right) // 2
        
        # 세 가지 경우 중 최대값
        return max(
            helper(arr, left, mid),           # 왼쪽 부분
            helper(arr, mid + 1, right),       # 오른쪽 부분
            max_crossing_sum(arr, left, mid, right)  # 중간 걸침
        )
    
    return helper(arr, 0, len(arr) - 1)

### 3. 백트래킹 (Backtracking)
```python
# 백트래킹 = 체계적인 탐색 + 가지치기
# 모든 가능한 경우를 탐색하되, 유망하지 않으면 즉시 포기

# 예제 1: 순열 생성 (Permutations)
def generate_permutations(nums):
    """모든 순열 생성
    시간복잡도: O(n! × n)
    공간복잡도: O(n)
    """
    def backtrack(path, remaining):
        # Base case: 모든 숫자를 사용함
        if not remaining:
            result.append(path[:])
            return
        
        # 남은 숫자들 중 하나씩 선택
        for i in range(len(remaining)):
            # 선택
            path.append(remaining[i])
            # 재귀 호출 (remaining에서 i번째 제외)
            backtrack(path, remaining[:i] + remaining[i+1:])
            # 백트래킹 (선택 취소)
            path.pop()
    
    result = []
    backtrack([], nums)
    return result

# 예제 2: 부분집합 생성 (Subsets)
def generate_subsets(nums):
    """모든 부분집합 생성
    시간복잡도: O(2^n × n)
    """
    def backtrack(start, path):
        # 현재 경로를 결과에 추가
        result.append(path[:])
        
        # start부터 끝까지 원소들을 하나씩 선택
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)  # 다음 원소부터 탐색
            path.pop()  # 백트래킹
    
    result = []
    backtrack(0, [])
    return result

# 예제 3: N-Queens 문제 (간단한 버전)
def solve_n_queens_simple(n):
    """N×N 체스판에 N개의 퀸을 서로 공격하지 않게 배치
    시간복잡도: O(n!)
    """
    def is_safe(board, row, col):
        # 같은 열에 퀸이 있는지 확인
        for i in range(row):
            if board[i] == col:
                return False
            # 대각선 확인
            if abs(board[i] - col) == abs(i - row):
                return False
        return True
    
    def backtrack(row):
        if row == n:
            # 해를 찾음
            solutions.append(board[:])
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col  # 퀸 배치
                backtrack(row + 1)  # 다음 행으로
                # board[row] = -1  # 명시적 백트래킹 (선택사항)
    
    board = [-1] * n  # board[i] = j는 i행 j열에 퀸 배치
    solutions = []
    backtrack(0)
    return solutions

# 고급: 백트래킹 최적화 기법
def sudoku_solver(board):
    """스도쿠 풀이 - 백트래킹 + 최적화
    
    최적화 기법:
    1. MRV (Minimum Remaining Values): 가능한 값이 가장 적은 칸부터
    2. Forward Checking: 선택 후 즉시 제약 확인
    3. Constraint Propagation: 제약 전파
    """
    def is_valid(board, row, col, num):
        # 행 체크
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # 열 체크
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # 3×3 박스 체크
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def find_empty_cell():
        """빈 칸 찾기 (MRV 휴리스틱 적용 가능)"""
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return None, None
    
    def solve():
        row, col = find_empty_cell()
        if row is None:
            return True  # 모든 칸이 채워짐
        
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                
                if solve():
                    return True
                
                board[row][col] = 0  # 백트래킹
        
        return False
    
    solve()
    return board

```

### 4. Binary Tree (이진 트리) 기초
```python
class TreeNode:
    """이진 트리 노드 클래스"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    solve()
    return board

```

### 4. Binary Tree (이진 트리) 기초
```python
class TreeNode:
    """이진 트리 노드 클래스"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 트리 순회의 4가지 방법

# 1. 전위 순회 (Preorder): 루트 → 왼쪽 → 오른쪽
def preorder_recursive(root):
    """전위 순회 - 재귀 버전
    시간복잡도: O(n)
    공간복잡도: O(h) - h는 트리 높이
    """
    if not root:
        return []
    
    result = [root.val]
    result.extend(preorder_recursive(root.left))
    result.extend(preorder_recursive(root.right))
    return result

# 2. 중위 순회 (Inorder): 왼쪽 → 루트 → 오른쪽
def inorder_recursive(root):
    """중위 순회 - BST에서 정렬된 순서로 방문
    BST에서 중위 순회하면 오름차순 정렬!
    """
    if not root:
        return []
    
    result = []
    result.extend(inorder_recursive(root.left))
    result.append(root.val)
    result.extend(inorder_recursive(root.right))
    return result

# 3. 후위 순회 (Postorder): 왼쪽 → 오른쪽 → 루트
def postorder_recursive(root):
    """후위 순회 - 삭제나 크기 계산에 유용
    자식을 모두 처리한 후 부모 처리
    """
    if not root:
        return []
    
    result = []
    result.extend(postorder_recursive(root.left))
    result.extend(postorder_recursive(root.right))
    result.append(root.val)
    return result

# 4. 레벨 순회 (Level-order): BFS, 각 레벨별로 순회
from collections import deque

def level_order(root):
    """레벨 순회 - BFS 사용
    시간복잡도: O(n)
    공간복잡도: O(w) - w는 최대 너비
    """
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

# 반복문으로 순회 구현 (스택 사용)
def preorder_iterative(root):
    """전위 순회 - 반복문 버전
    스택을 사용하여 재귀 시뮬레이션
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # 오른쪽을 먼저 넣어야 왼쪽이 먼저 처리됨
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

def inorder_iterative(root):
    """중위 순회 - 반복문 버전
    왼쪽 끝까지 간 후 처리하고 오른쪽으로
    """
    result = []
    stack = []
    current = root
    
    while stack or current:
        # 왼쪽 끝까지 이동
        while current:
            stack.append(current)
            current = current.left
        
        # 노드 처리
        current = stack.pop()
        result.append(current.val)
        
        # 오른쪽으로 이동
        current = current.right
    
    return result

# 중급: 트리의 속성 계산
def tree_height(root):
    """트리의 높이(깊이) 계산
    시간복잡도: O(n)
    공간복잡도: O(h)
    """
    if not root:
        return 0
    
    left_height = tree_height(root.left)
    right_height = tree_height(root.right)
    
    return 1 + max(left_height, right_height)

def count_nodes(root):
    """트리의 노드 개수 계산
    시간복잡도: O(n)
    """
    if not root:
        return 0
    
    return 1 + count_nodes(root.left) + count_nodes(root.right)

def is_balanced(root):
    """균형 이진 트리 확인
    모든 노드의 왼쪽/오른쪽 서브트리 높이 차이가 1 이하
    """
    def check_height(node):
        if not node:
            return 0
        
        left_height = check_height(node.left)
        if left_height == -1:
            return -1
        
        right_height = check_height(node.right)
        if right_height == -1:
            return -1
        
        if abs(left_height - right_height) > 1:
            return -1
        
        return max(left_height, right_height) + 1
    
    return check_height(root) != -1

# 고급: Morris Traversal (O(1) 공간복잡도)
def morris_inorder(root):
    """Morris 중위 순회 - 스택 없이 순회
    공간복잡도: O(1)
    시간복잡도: O(n)
    
    아이디어: 임시로 트리 구조를 수정하여 순회
    """
    result = []
    current = root
    
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            # 왼쪽 서브트리의 가장 오른쪽 노드 찾기
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # 임시 연결 생성
                predecessor.right = current
                current = current.left
            else:
                # 원상 복구
                predecessor.right = None
                result.append(current.val)
                current = current.right
    
    return result

# 고급: 트리 재구성
def build_tree_from_traversals(preorder, inorder):
    """전위 + 중위 순회 결과로 트리 재구성
    시간복잡도: O(n)
    공간복잡도: O(n)
    """
    if not preorder or not inorder:
        return None
    
    # 전위 순회의 첫 번째는 항상 루트
    root = TreeNode(preorder[0])
    
    # 중위 순회에서 루트의 위치 찾기
    mid = inorder.index(preorder[0])
    
    # 재귀적으로 왼쪽/오른쪽 서브트리 구성
    root.left = build_tree_from_traversals(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree_from_traversals(preorder[mid+1:], inorder[mid+1:])
    
    return root
```

## 💡 주요 패턴

### 패턴 1: 재귀 함수 설계 패턴
- **사용 상황**: 트리/그래프 탐색, 분할 정복, 백트래킹
- **시간복잡도**: 재귀 호출 횟수 × 각 호출의 작업량
- **공간복잡도**: O(재귀 깊이) - 콜 스택

```python
# 재귀 설계 템플릿
def recursive_template(data, current_state):
    """재귀 함수 설계 템플릿"""
    # 1. 베이스 케이스 체크
    if is_base_case(data):
        return base_result
    
    # 2. 현재 노드/상태 처리
    process_current(current_state)
    
    # 3. 재귀 호출
    result = []
    for next_state in get_next_states(data):
        result.append(recursive_template(data, next_state))
    
    # 4. 결과 조합 및 반환
    return combine_results(result)

# Helper 함수 패턴 - 추가 파라미터가 필요한 경우
def main_function(data):
    """메인 함수 - 외부 인터페이스"""
    def helper(data, param1, param2):
        """헬퍼 함수 - 실제 재귀 로직"""
        # 베이스 케이스
        if not data:
            return result
        
        # 재귀 처리
        # ...
        return helper(modified_data, new_param1, new_param2)
    
    # 초기 호출
    return helper(data, initial_param1, initial_param2)

# 재귀 트리 분석 예제
def count_nodes(root):
    """이진 트리의 노드 개수 계산
    시간복잡도: O(n) - 모든 노드 방문
    공간복잡도: O(h) - h는 트리 높이"""
    if not root:
        return 0
    
    # 현재 노드(1) + 왼쪽 서브트리 + 오른쪽 서브트리
    return 1 + count_nodes(root.left) + count_nodes(root.right)

def max_depth(root):
    """이진 트리의 최대 깊이
    시간복잡도: O(n)
    공간복잡도: O(h)"""
    if not root:
        return 0
    
    # 왼쪽과 오른쪽 중 더 깊은 쪽 + 1
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### 패턴 2: 트리 순회 패턴
- **사용 상황**: 트리 탐색, 경로 찾기, 트리 변환
- **시간복잡도**: O(n) - 모든 노드 방문
- **공간복잡도**: O(h) 재귀, O(n) 반복문

```python
# DFS 순회 패턴 (재귀)
def dfs_traversal(root):
    """DFS 순회 기본 템플릿"""
    def dfs(node, path):
        if not node:
            return
        
        # 전위 처리 위치
        path.append(node.val)
        
        # 왼쪽 서브트리 탐색
        dfs(node.left, path)
        
        # 중위 처리 위치
        
        # 오른쪽 서브트리 탐색
        dfs(node.right, path)
        
        # 후위 처리 위치
        path.pop()  # 백트래킹
    
    result = []
    dfs(root, result)
    return result

# 경로 추적 패턴
def has_path_sum(root, target_sum):
    """루트에서 리프까지 경로 합이 target_sum인지 확인"""
    if not root:
        return False
    
    # 리프 노드 도달
    if not root.left and not root.right:
        return root.val == target_sum
    
    # 남은 합으로 재귀 호출
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or 
            has_path_sum(root.right, remaining))

def binary_tree_paths(root):
    """루트에서 리프까지 모든 경로 반환"""
    def dfs(node, path, paths):
        if not node:
            return
        
        # 현재 노드를 경로에 추가
        path.append(str(node.val))
        
        # 리프 노드인 경우
        if not node.left and not node.right:
            paths.append('->'.join(path))
        else:
            # 계속 탐색
            dfs(node.left, path, paths)
            dfs(node.right, path, paths)
        
        # 백트래킹
        path.pop()
    
    paths = []
    dfs(root, [], paths)
    return paths

# Bottom-up 패턴 (후위 순회 활용)
def is_balanced(root):
    """균형 이진 트리 확인
    각 노드의 왼쪽/오른쪽 서브트리 높이 차이가 1 이하"""
    def check_height(node):
        if not node:
            return 0
        
        # 왼쪽 서브트리 확인
        left_height = check_height(node.left)
        if left_height == -1:
            return -1
        
        # 오른쪽 서브트리 확인
        right_height = check_height(node.right)
        if right_height == -1:
            return -1
        
        # 현재 노드의 균형 확인
        if abs(left_height - right_height) > 1:
            return -1
        
        return max(left_height, right_height) + 1
    
    return check_height(root) != -1
```

## 🔑 Python 필수 문법

### 자료구조 관련
```python
# TreeNode 클래스 활용
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        """디버깅을 위한 문자열 표현"""
        return f"TreeNode({self.val})"

# 트리 생성 헬퍼 함수
def build_tree(values):
    """리스트로부터 이진 트리 생성 (레벨 순서)"""
    if not values:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        # 왼쪽 자식
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        # 오른쪽 자식
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root

# 재귀 제한 설정
import sys
sys.setrecursionlimit(10**6)  # 기본값 1000에서 증가

# 중첩 함수와 nonlocal
def outer_function():
    count = 0  # 외부 함수의 변수
    
    def inner_function():
        nonlocal count  # 외부 변수 수정 가능
        count += 1
        return count
    
    return inner_function

# 함수 내 함수 (클로저)
def create_counter():
    """클로저를 이용한 카운터"""
    count = 0
    
    def increment():
        nonlocal count
        count += 1
        return count
    
    return increment

counter = create_counter()
print(counter())  # 1
print(counter())  # 2
```

### 유용한 메서드/함수
```python
# collections.deque - 레벨 순회에 필수
from collections import deque

queue = deque([1, 2, 3])
queue.append(4)      # 오른쪽 추가
queue.appendleft(0)  # 왼쪽 추가
queue.popleft()      # 왼쪽 제거 O(1)

# 재귀 함수 디버깅
def debug_recursion(func):
    """재귀 함수 디버깅용 데코레이터"""
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"Call {wrapper.calls}: {func.__name__}{args}")
        result = func(*args, **kwargs)
        print(f"Return: {result}")
        return result
    wrapper.calls = 0
    return wrapper

@debug_recursion
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# 메모이제이션 미리보기 (Week 5에서 자세히)
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memo(n):
    """메모이제이션으로 최적화된 피보나치
    시간복잡도: O(n)
    공간복잡도: O(n)"""
    if n <= 1:
        return n
    return fibonacci_memo(n-1) + fibonacci_memo(n-2)

# 트리 시각화 (간단한 출력)
def print_tree(root, level=0, prefix="Root: "):
    """트리를 시각적으로 출력"""
    if root:
        print(" " * (level * 4) + prefix + str(root.val))
        if root.left or root.right:
            if root.left:
                print_tree(root.left, level + 1, "L--- ")
            else:
                print(" " * ((level + 1) * 4) + "L--- None")
            if root.right:
                print_tree(root.right, level + 1, "R--- ")
            else:
                print(" " * ((level + 1) * 4) + "R--- None")
```

## 🎯 LeetCode 추천 문제

### 필수 문제
- [ ] [100] Same Tree - 트리 비교 기초
- [ ] [101] Symmetric Tree - 대칭 트리 확인
- [ ] [104] Maximum Depth of Binary Tree - 트리 높이
- [ ] [22] Generate Parentheses - 백트래킹 기초
- [ ] [226] Invert Binary Tree - 트리 변환

### 도전 문제
- [ ] [199] Binary Tree Right Side View - 레벨 순회 응용
- [ ] [110] Balanced Binary Tree - Bottom-up 재귀
- [ ] [236] Lowest Common Ancestor - LCA 찾기

### 추가 연습
- [ ] [111] Minimum Depth of Binary Tree
- [ ] [112] Path Sum
- [ ] [257] Binary Tree Paths
- [ ] [94] Binary Tree Inorder Traversal
- [ ] [144] Binary Tree Preorder Traversal
- [ ] [145] Binary Tree Postorder Traversal