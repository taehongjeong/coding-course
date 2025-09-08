# Week 05: Tree 심화 & BST (Binary Search Tree)

## 📖 핵심 개념

### 1. Tree DFS/BFS 심화 탐색
```python
# Tree DFS 심화: Top-down vs Bottom-up 접근법
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 1. Top-down 접근: 부모에서 자식으로 정보 전달
def max_depth_topdown(root):
    """Top-down: 현재 깊이를 매개변수로 전달
    시간복잡도: O(n)
    공간복잡도: O(h) - h는 트리 높이
    """
    def dfs(node, depth):
        if not node:
            return depth
        
        # 현재 깊이를 자식에게 전달
        left_depth = dfs(node.left, depth + 1)
        right_depth = dfs(node.right, depth + 1)
        
        return max(left_depth, right_depth)
    
    return dfs(root, 0)

# 2. Bottom-up 접근: 자식에서 부모로 정보 수집
def max_depth_bottomup(root):
    """Bottom-up: 자식의 결과를 수집해서 계산
    시간복잡도: O(n)
    공간복잡도: O(h)
    """
    if not root:
        return 0
    
    # 자식들의 결과를 먼저 얻음
    left_depth = max_depth_bottomup(root.left)
    right_depth = max_depth_bottomup(root.right)
    
    # 결과를 이용해 현재 노드 계산
    return 1 + max(left_depth, right_depth)

# 3. 경로 탐색 심화: 모든 경로 찾기
def find_all_paths(root, target):
    """목표값까지의 모든 경로 찾기
    시간복잡도: O(n²) - 최악의 경우
    공간복잡도: O(n)
    """
    def dfs(node, remaining, path, all_paths):
        if not node:
            return
        
        # 현재 노드를 경로에 추가
        path.append(node.val)
        remaining -= node.val
        
        # 리프 노드에서 목표값 확인
        if not node.left and not node.right and remaining == 0:
            all_paths.append(path[:])  # 복사본 추가
        
        # 자식 탐색
        dfs(node.left, remaining, path, all_paths)
        dfs(node.right, remaining, path, all_paths)
        
        # 백트래킹
        path.pop()
    
    all_paths = []
    dfs(root, target, [], all_paths)
    return all_paths

# 4. BFS 심화: 레벨별 처리
from collections import deque

def zigzag_level_order(root):
    """지그재그 레벨 순회
    시간복잡도: O(n)
    공간복잡도: O(w) - w는 최대 너비
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            # 방향에 따라 다르게 추가
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(level))
        left_to_right = not left_to_right
    
    return result

# 5. 서브트리 문제 패턴
def is_subtree(root, subRoot):
    """서브트리 확인
    시간복잡도: O(m × n)
    공간복잡도: O(max(m, n))
    """
    def is_same_tree(p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        return (p.val == q.val and 
                is_same_tree(p.left, q.left) and 
                is_same_tree(p.right, q.right))
    
    if not root:
        return False
    
    # 현재 노드부터 같은지 확인
    if is_same_tree(root, subRoot):
        return True
    
    # 왼쪽이나 오른쪽 서브트리에 있는지 확인
    return is_subtree(root.left, subRoot) or is_subtree(root.right, subRoot)

# 6. 조상(Ancestor) 문제 패턴
def lowest_common_ancestor(root, p, q):
    """최소 공통 조상 찾기
    시간복잡도: O(n)
    공간복잡도: O(h)
    """
    # Base case
    if not root or root == p or root == q:
        return root
    
    # 왼쪽과 오른쪽 서브트리 탐색
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    # 양쪽에서 찾았다면 현재 노드가 LCA
    if left and right:
        return root
    
    # 한쪽에서만 찾았다면 그쪽이 LCA
    return left if left else right
```

### 2. BST (Binary Search Tree) 완전 정복
```python
# BST의 핵심 속성
# 1. 왼쪽 서브트리의 모든 노드 < 현재 노드
# 2. 오른쪽 서브트리의 모든 노드 > 현재 노드
# 3. 중위 순회 시 오름차순 정렬

class BSTNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 1. BST 검색
def search_bst(root, target):
    """BST에서 값 검색
    시간복잡도: O(h) - 균형 시 O(log n), 최악 O(n)
    공간복잡도: O(1) 반복, O(h) 재귀
    """
    # 반복문 버전 (더 효율적)
    current = root
    while current:
        if current.val == target:
            return current
        elif target < current.val:
            current = current.left
        else:
            current = current.right
    return None
    
    # 재귀 버전
    # if not root or root.val == target:
    #     return root
    # if target < root.val:
    #     return search_bst(root.left, target)
    # return search_bst(root.right, target)

# 2. BST 삽입
def insert_bst(root, val):
    """BST에 새 값 삽입
    시간복잡도: O(h)
    공간복잡도: O(1) 반복, O(h) 재귀
    """
    if not root:
        return BSTNode(val)
    
    # 반복문 버전
    current = root
    while True:
        if val < current.val:
            if current.left:
                current = current.left
            else:
                current.left = BSTNode(val)
                break
        else:
            if current.right:
                current = current.right
            else:
                current.right = BSTNode(val)
                break
    
    return root
    
    # 재귀 버전
    # if not root:
    #     return BSTNode(val)
    # if val < root.val:
    #     root.left = insert_bst(root.left, val)
    # else:
    #     root.right = insert_bst(root.right, val)
    # return root

# 3. BST 삭제 (가장 복잡한 연산)
def delete_bst(root, key):
    """BST에서 노드 삭제
    시간복잡도: O(h)
    공간복잡도: O(h)
    
    3가지 경우:
    1. 리프 노드: 그냥 삭제
    2. 자식이 하나: 자식으로 대체
    3. 자식이 둘: 후속자(successor)로 대체
    """
    if not root:
        return None
    
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        # 삭제할 노드를 찾음
        
        # Case 1: 리프 노드
        if not root.left and not root.right:
            return None
        
        # Case 2: 자식이 하나
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        
        # Case 3: 자식이 둘
        # 오른쪽 서브트리의 최솟값(후속자) 찾기
        min_node = root.right
        while min_node.left:
            min_node = min_node.left
        
        # 현재 노드의 값을 후속자 값으로 대체
        root.val = min_node.val
        
        # 오른쪽 서브트리에서 후속자 삭제
        root.right = delete_bst(root.right, min_node.val)
    
    return root

# 4. BST 유효성 검증
def is_valid_bst(root):
    """유효한 BST인지 확인
    시간복잡도: O(n)
    공간복잡도: O(h)
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        # 현재 노드가 범위를 벗어나면 False
        if node.val <= min_val or node.val >= max_val:
            return False
        
        # 왼쪽은 현재값보다 작아야 하고
        # 오른쪽은 현재값보다 커야 함
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

# 5. BST에서 k번째 작은 원소
def kth_smallest(root, k):
    """k번째 작은 원소 찾기
    시간복잡도: O(h + k)
    공간복잡도: O(h)
    """
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    # 방법 1: 전체 중위 순회
    # return inorder(root)[k-1]
    
    # 방법 2: k개만 찾고 중단 (더 효율적)
    def inorder_k(node):
        if node:
            # 왼쪽 서브트리 탐색
            yield from inorder_k(node.left)
            # 현재 노드 방문
            yield node.val
            # 오른쪽 서브트리 탐색
            yield from inorder_k(node.right)
    
    gen = inorder_k(root)
    for _ in range(k-1):
        next(gen)
    return next(gen)

# 6. BST를 균형 BST로 변환
def balance_bst(root):
    """불균형 BST를 균형 BST로 변환
    시간복잡도: O(n)
    공간복잡도: O(n)
    """
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    def build_balanced_bst(nums, start, end):
        if start > end:
            return None
        
        mid = (start + end) // 2
        node = TreeNode(nums[mid])
        node.left = build_balanced_bst(nums, start, mid - 1)
        node.right = build_balanced_bst(nums, mid + 1, end)
        return node
    
    # 1. 중위 순회로 정렬된 배열 얻기
    nums = inorder(root)
    
    # 2. 정렬된 배열로 균형 BST 만들기
    return build_balanced_bst(nums, 0, len(nums) - 1)

# 7. BST Iterator (중위 순회 반복자)
class BSTIterator:
    """BST 반복자 - 중위 순회 순서로 원소 반환
    next(): O(1) amortized
    hasNext(): O(1)
    공간복잡도: O(h)
    """
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node):
        """왼쪽 경로의 모든 노드를 스택에 추가"""
        while node:
            self.stack.append(node)
            node = node.left
    
    def next(self):
        """다음 작은 수 반환"""
        node = self.stack.pop()
        if node.right:
            self._push_left(node.right)
        return node.val
    
    def hasNext(self):
        """다음 원소가 있는지 확인"""
        return len(self.stack) > 0
```

### 3. 트리 문제의 메모이제이션 활용
```python
from functools import lru_cache

# 1. 서브트리 합 메모이제이션
class Solution:
    def subtree_sum_frequency(self, root):
        """각 서브트리 합의 빈도수 계산
        시간복잡도: O(n)
        공간복잡도: O(n)
        """
        freq = defaultdict(int)
        
        def get_sum(node):
            if not node:
                return 0
            
            # 현재 서브트리의 합 계산
            subtree_sum = node.val + get_sum(node.left) + get_sum(node.right)
            freq[subtree_sum] += 1
            
            return subtree_sum
        
        get_sum(root)
        
        # 가장 빈번한 서브트리 합 찾기
        max_freq = max(freq.values())
        return [sum_val for sum_val, f in freq.items() if f == max_freq]

# 2. 동적 프로그래밍을 이용한 트리 문제
def house_robber_iii(root):
    """트리에서 인접하지 않은 노드의 최대 합
    시간복잡도: O(n)
    공간복잡도: O(h)
    """
    @lru_cache(maxsize=None)
    def rob_helper(node):
        if not node:
            return 0, 0  # (rob_root, not_rob_root)
        
        left_rob, left_not_rob = rob_helper(node.left)
        right_rob, right_not_rob = rob_helper(node.right)
        
        # 현재 노드를 털 경우: 자식은 털지 않음
        rob_root = node.val + left_not_rob + right_not_rob
        
        # 현재 노드를 털지 않을 경우: 자식은 털어도 되고 안 털어도 됨
        not_rob_root = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
        
        return rob_root, not_rob_root
    
    return max(rob_helper(root))

# 3. 경로 문제 메모이제이션
def unique_paths_in_tree(root):
    """트리에서 고유한 경로 개수 (메모이제이션 활용)
    시간복잡도: O(n)
    공간복잡도: O(n)
    """
    @lru_cache(maxsize=None)
    def count_paths(node, visited_values):
        if not node:
            return 0
        
        # 현재 값이 이미 방문한 값인지 확인
        if node.val in visited_values:
            return 0
        
        # 리프 노드면 경로 하나 완성
        if not node.left and not node.right:
            return 1
        
        # 현재 값을 방문 목록에 추가
        new_visited = visited_values | {node.val}
        
        # 왼쪽과 오른쪽 서브트리의 경로 수 합산
        left_paths = count_paths(node.left, new_visited)
        right_paths = count_paths(node.right, new_visited)
        
        return left_paths + right_paths
    
    return count_paths(root, frozenset())

# 4. 트리 직경 (가장 긴 경로)
def diameter_of_tree(root):
    """트리의 직경 (두 노드 간 가장 긴 경로)
    시간복잡도: O(n)
    공간복잡도: O(h)
    """
    max_diameter = 0
    
    def height(node):
        nonlocal max_diameter
        if not node:
            return 0
        
        # 왼쪽과 오른쪽 서브트리의 높이
        left_height = height(node.left)
        right_height = height(node.right)
        
        # 현재 노드를 통과하는 경로의 길이
        max_diameter = max(max_diameter, left_height + right_height)
        
        # 현재 노드의 높이 반환
        return 1 + max(left_height, right_height)
    
    height(root)
    return max_diameter

# 5. 트리 경로 합 캐싱
class PathSumCache:
    """경로 합 계산을 캐싱하는 클래스"""
    def __init__(self):
        self.cache = {}
    
    def path_sum_equals(self, root, target_sum):
        """캐싱을 이용한 경로 합 확인
        시간복잡도: O(n²) worst, O(n) with cache
        공간복잡도: O(n)
        """
        def dfs(node, current_path, current_sum):
            if not node:
                return 0
            
            # 캐시 키 생성
            cache_key = (id(node), tuple(current_path), current_sum)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            current_sum += node.val
            current_path.append(node.val)
            
            # 현재 경로가 목표 합과 같은지 확인
            count = 1 if current_sum == target_sum else 0
            
            # 중간부터 시작하는 경로 확인
            temp_sum = current_sum
            for i in range(len(current_path) - 1):
                temp_sum -= current_path[i]
                if temp_sum == target_sum:
                    count += 1
            
            # 자식 노드 탐색
            count += dfs(node.left, current_path[:], current_sum)
            count += dfs(node.right, current_path[:], current_sum)
            
            # 결과 캐싱
            self.cache[cache_key] = count
            return count
        
        return dfs(root, [], 0)
```

## 💡 주요 패턴

### 패턴 1: Top-down vs Bottom-up DFS
- **사용 상황**: 트리 정보 전달 방향에 따라 선택
- **시간복잡도**: 둘 다 O(n)
- **공간복잡도**: O(h) - 재귀 스택

```python
# Top-down 패턴: 부모 → 자식 정보 전달
def top_down_pattern(root):
    """부모에서 자식으로 정보를 전달하며 처리"""
    def dfs(node, parent_info):
        if not node:
            return
        
        # 부모로부터 받은 정보로 현재 노드 처리
        current_info = process(parent_info, node.val)
        
        # 자식에게 정보 전달
        dfs(node.left, current_info)
        dfs(node.right, current_info)
    
    dfs(root, initial_info)

# Bottom-up 패턴: 자식 → 부모 정보 수집
def bottom_up_pattern(root):
    """자식의 결과를 수집하여 부모에서 처리"""
    def dfs(node):
        if not node:
            return base_value
        
        # 자식들의 결과 먼저 수집
        left_result = dfs(node.left)
        right_result = dfs(node.right)
        
        # 수집한 정보로 현재 노드 처리
        return process(node.val, left_result, right_result)
    
    return dfs(root)
```

### 패턴 2: BST 연산 패턴
- **사용 상황**: 정렬된 데이터의 효율적 관리
- **시간복잡도**: O(h) - 균형 시 O(log n)
- **공간복잡도**: O(1) 반복, O(h) 재귀

```python
# BST 탐색 패턴
def bst_operation_pattern(root, target):
    """BST 속성을 활용한 효율적 탐색"""
    current = root
    
    while current:
        if current.val == target:
            # 목표 찾음
            return current
        elif target < current.val:
            # 왼쪽 서브트리로
            current = current.left
        else:
            # 오른쪽 서브트리로
            current = current.right
    
    return None

# BST 범위 탐색 패턴
def range_search_bst(root, low, high):
    """BST에서 범위 내 값들 찾기"""
    result = []
    
    def dfs(node):
        if not node:
            return
        
        # 가지치기: BST 속성 활용
        if node.val > low:
            dfs(node.left)  # 왼쪽에 더 작은 값 있을 수 있음
        
        if low <= node.val <= high:
            result.append(node.val)
        
        if node.val < high:
            dfs(node.right)  # 오른쪽에 더 큰 값 있을 수 있음
    
    dfs(root)
    return result
```

## 🔑 Python 필수 문법

### 자료구조 관련
```python
# BST 노드 클래스
class BSTNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"BSTNode({self.val})"
    
    def __lt__(self, other):
        """비교 연산자 오버로딩 (heapq 사용 시)"""
        return self.val < other.val

# 제너레이터를 이용한 중위 순회
def inorder_generator(root):
    """메모리 효율적인 중위 순회"""
    if root:
        yield from inorder_generator(root.left)
        yield root.val
        yield from inorder_generator(root.right)

# 사용 예
for val in inorder_generator(root):
    print(val)  # 정렬된 순서로 출력

# nonlocal 키워드 활용
def tree_problem():
    """외부 변수 수정이 필요한 경우"""
    max_value = float('-inf')
    
    def dfs(node):
        nonlocal max_value  # 외부 변수 수정 가능
        if not node:
            return
        
        max_value = max(max_value, node.val)
        dfs(node.left)
        dfs(node.right)
    
    dfs(root)
    return max_value
```

### 유용한 메서드/함수
```python
# functools.lru_cache - 메모이제이션
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_tree_operation(node_id, param):
    """캐싱으로 중복 계산 방지
    - maxsize=None: 무제한 캐시
    - maxsize=128: 최근 128개만 캐시 (기본값)
    """
    # 복잡한 계산...
    return result

# 캐시 관리
expensive_tree_operation.cache_info()  # 캐시 통계
expensive_tree_operation.cache_clear()  # 캐시 초기화

# defaultdict로 트리 레벨별 노드 관리
from collections import defaultdict

def level_nodes(root):
    """레벨별 노드 그룹화"""
    levels = defaultdict(list)
    
    def dfs(node, level):
        if not node:
            return
        levels[level].append(node.val)
        dfs(node.left, level + 1)
        dfs(node.right, level + 1)
    
    dfs(root, 0)
    return dict(levels)

# bisect로 BST 값 찾기 (정렬된 리스트에서)
from bisect import bisect_left, insort

class SortedBST:
    """정렬된 리스트로 BST 시뮬레이션"""
    def __init__(self):
        self.values = []
    
    def insert(self, val):
        """O(n) 삽입 but 간단한 구현"""
        insort(self.values, val)
    
    def search(self, val):
        """O(log n) 검색"""
        idx = bisect_left(self.values, val)
        return idx < len(self.values) and self.values[idx] == val
    
    def kth_smallest(self, k):
        """O(1) k번째 원소"""
        return self.values[k-1] if k <= len(self.values) else None

# yield from을 이용한 재귀 제너레이터
def all_paths(root):
    """모든 루트-리프 경로 생성"""
    def dfs(node, path):
        if not node:
            return
        
        path.append(node.val)
        
        if not node.left and not node.right:
            yield path[:]  # 경로 복사본 yield
        else:
            yield from dfs(node.left, path)
            yield from dfs(node.right, path)
        
        path.pop()
    
    yield from dfs(root, [])

# 사용
for path in all_paths(root):
    print("경로:", "->".join(map(str, path)))
```

## 🎯 LeetCode 추천 문제

### 필수 문제
- [ ] [98] Validate Binary Search Tree - BST 유효성 검증
- [ ] [108] Convert Sorted Array to Binary Search Tree - 정렬 배열 → BST
- [ ] [230] Kth Smallest Element in a BST - k번째 작은 원소
- [ ] [235] Lowest Common Ancestor of a BST - BST에서 LCA
- [ ] [450] Delete Node in a BST - BST 노드 삭제

### 도전 문제
- [ ] [669] Trim a Binary Search Tree - BST 범위 잘라내기
- [ ] [938] Range Sum of BST - BST 범위 합
- [ ] [173] Binary Search Tree Iterator - BST 반복자
- [ ] [337] House Robber III - 트리 DP

### 추가 연습
- [ ] [700] Search in a Binary Search Tree - BST 검색
- [ ] [701] Insert into a Binary Search Tree - BST 삽입
- [ ] [653] Two Sum IV - Input is a BST
- [ ] [102] Binary Tree Level Order Traversal
- [ ] [103] Binary Tree Zigzag Level Order Traversal
- [ ] [543] Diameter of Binary Tree