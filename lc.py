class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # O(n) complexity solutions
        num_map = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map:
                return [num_map[complement], i]
            num_map[num] = i
       
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy_head = ListNode(0)
        current = dummy_head  # pointer to base list
        carry = 0 # carry if addition result > 9

        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            total = val1 + val2 + carry
            carry = total // 10
            current.next = ListNode(total % 10)
            current = current.next

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy_head.next

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        char_map = {}
        max_length = 0 
        start = 0 
        for i, char in enumerate(s):
            if char in char_map and char_map[char] >= start:
                start = char_map[char] + 1

            char_map[char] = i
            max_length = max(max_length, i - start + 1)
        return max_length

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman_map = {'I':1, 
                     'V':5,
                     'X':10,
                     'L':50,
                     'C':100,
                     'D':500,
                     'M':1000}
        result = 0
        n = len(s)
        for i in range(n):
            ### HOW TO AVOID OUT OF BOUNCE !!! ###
            if i < n-1 and roman_map[s[i]] < roman_map[s[i+1]]:
                result -= roman_map[s[i]]
            else:
                result += roman_map[s[i]]
        return result
    
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
        
        prefix = strs[0]
        for i in range(1, len(strs)):
            while strs[i].find(prefix) != 0:
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
    
    ## Time limit exceded xD
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = len(s)
        pal_list = []
        for i in range(l):
            for j in range(i+1, l+1):
                sub = s[i:j]
                if sub == sub[::-1]:
                    pal_list.append(sub)

        max_len = -1
        res = ""
        for e in pal_list:
            if len(e) > max_len:
                max_len = len(e)
                res = e
        return res

    # optimized version
    def longestPalindrome_2(self, s):
        def expand_around_center(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
                print("left: ", left)
                print("right:", right)
            return s[left:1+right]

        if len(s) == 0:
            return ""
        
        longest_palindrome = ""
        for i in range(len(s)):

            odd_palindrome = expand_around_center(i, i)
            even_panlindrome = expand_around_center(i, i+1)
            longer_palindrome = odd_palindrome if len(odd_palindrome) > len(even_panlindrome) else even_panlindrome

            if len(longer_palindrome) > len(longest_palindrome):
                longest_palindrome = longer_palindrome
        
        return longest_palindrome
    
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or numRows >= len(s):
            return s
        
        rows = [''] * numRows
        current_row = 0
        going_down = False
        
        for char in s:
            rows[current_row] += char
            if current_row == 0 or current_row == numRows - 1:
                going_down = not going_down
            current_row += 1 if going_down else -1
        
        return ''.join(rows)

    def isValid(self, s):
        bracket_map = {')': '(', ']': '[', '}': '{'}
        stack = []
        
        for char in s:
            if char in bracket_map: 
                top_element = stack.pop() if stack else "#"
                if bracket_map[char] != top_element:
                    return False
            else: 
                stack.append(char)
                print(stack)
        
        return not stack
    
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        result = []
        
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            left, right = i + 1, len(nums) - 1
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if current_sum == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                        
                    left += 1
                    right -= 1
                    
                elif current_sum < 0:
                    left += 1 
                else:
                    right -= 1 
        return result
    
    
    def reverse_integer(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = -1 if x < 0 else 1
        x = abs(x)
        
        revers = int(str(x)[::-1])
        revers *= sign
        
        if revers < -2**31 or revers > 2**31 - 1:
            return 0
        
        return revers
    
    def mergeTwoLists(self, l1, l2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy_head = ListNode(0)
        current = dummy_head

        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next

            current = current.next
        
        if l1:
            current.next = l1
        elif l2:
            current.next = l2
        
        return dummy_head.next
    
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.strip()
        if not s:
            return 0
        
        sign = 1
        index = 0

        if s[0] == "-":
            sign = -1
            index += 1
        elif s[0] == '+':
            index += 1
        
        num = 0
        while index < len(s) and s[index].isdigit():
            num = num*10 + int(s[index])
            index += 1
        
        num *= sign
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31

        if num > INT_MAX:
            return INT_MAX
        if num < INT_MIN:
            return INT_MIN
              
        return num
    
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        j = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[j]:
                j+=1
                nums[j] = nums[i]

        return j + 1
    
    ### random test - inverting variables with xor ###
    def invert_vars_with_xor():
        a = "{0:08b}".format(10)
        b = "{0:08b}".format(12)

        a_int = int(a, 2)
        b_int = int(b, 2)

        print("a", a)
        print("b", b)

        c = a_int ^ b_int
        c_binary = "{0:08b}".format(c)

        print("c", c_binary)

        d = c ^ b_int
        d_binary = "{0:08b}".format(d)
        print("d which is (a)", d_binary)

        e = c ^ a_int
        e_binary = "{0:08b}".format(e)
        print("e which is (b)", e_binary)
    
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        # modification should be done "in-place"
        if not nums:
            return 0
            
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k
    
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        val = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I')
        ]
        roman_int = ""
        for value, roman in val:
            while num >= value:
                roman_int += roman
                num -= value

        return roman_int
    
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle not in haystack:
            return -1
        
        l = len(haystack)
        i = 0
        if needle in haystack:
            i = haystack.index(needle)
            
        return i
    
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        closest_sum = 2**31 - 1
        
        for i in range(len(nums)-2):
            left = i + 1
            right = len(nums) - 1
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                    
                if current_sum == target:
                    return current_sum
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
                
        return closest_sum
    
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # Two-Pointer approach
        left = 0
        right = len(height) - 1
        max_area = 0
        
        while left < right:
            curr_area = min(height[left], height[right]) * (right - left)
            max_area = max(max_area, curr_area)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
            
        return max_area
    
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        ## Run-time must be O(log n)
        i = 0
        j = len(nums) - 1
        
        while i <= j:
            x = (i+j) // 2
            if nums[x] == target:
                return x
            elif nums[x] < target:
                i = x + 1      
            else:
                j = x - 1
            
        return i
    
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        # Two-pointer approach
        dummy = ListNode(0, head)
        first = dummy
        second = dummy
        
        for _ in range(n+1):
            first = first.next
        
        while first is not None:
            first = first.next
            second = second.next
        
        second.next = second.next.next
        
        return dummy.next
    
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if digits == "":
            return []
        
        number_map = {"2" : "abc",
                     "3": "def",
                     "4": "ghi",
                     "5": "jkl",
                     "6": "mno",
                     "7": "pqrs",
                     "8": "tuv",
                     "9": "wxyz"}
        
        char_list = [""]
        for digit in digits:
            digit_to_chars = []
            for product in char_list:
                for char in number_map[digit]:
                    digit_to_chars.append(product + char)
                    
            char_list = digit_to_chars
        
        return char_list

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def isValid(s):
            bracket_map = {')': '('}
            stack = []

            for char in s:
                if char in bracket_map: 
                    top_element = stack.pop() if stack else "#"
                    if bracket_map[char] != top_element:
                        return False
                else: 
                    stack.append(char)

            return not stack
        
        res = []
        
        def backtrack(open, close, current):
            if len(current) == 2*n:
                res.append(current)
                return
            if open < n:
                backtrack(open+1, close, current+'(')
            if close < open:
                backtrack(open, close+1, current+')')
        
        backtrack(0, 0, "")
        
        for i in res:
            if isValid(i):
                continue
            else:
                res.remove(i)
        
        return res
    
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums.sort()
        result = []
        l = len(nums)
        
        for i in range(l-3):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            for j in range(i+1, l-2):
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                
                left, right = j+1, l-1
                
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    
                    if current_sum == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                    
                        left += 1
                        right -= 1
                    
                    elif current_sum < target:
                        left += 1
                    else:
                        right -= 1
                        
        return result
    
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        i = l-2
        while i >= 0 and nums[i] >= nums[i+1]:
            i -= 1
        if i >= 0:
            j = l-1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
            
        nums[i + 1:] = reversed(nums[i + 1:])

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # Must be O(log n) -> Binary Search
        def binarySearchLeft(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left
        
        def binarySearchRight(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid - 1
            return right
        
        left_index = binarySearchLeft(nums, target)
        right_index = binarySearchRight(nums, target)

        if left_index < right_index and right_index < len(nums) and nums[left_index] == target and nums[right_index] == target:
            return [left_index, right_index]
        else:
            return [-1, -1]
    
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        previous = dummy
        
        while head and head.next:
            first = head
            second = head.next
            
            previous.next = second
            first.next = second.next
            second.next = first
            
            previous = first
            head = first.next
    
        return dummy.next
    
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
    
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        base = "1"
        for i in range(n - 1):
            temp = base
            base = ""
            result = temp[0]
            count = 1
            for j in range(1, len(temp)):
                if temp[j] != temp[j-1]:
                    base += str(count)
                    base += result
                    result = temp[j]
                    count = 0
                count += 1
            base += str(count)
            base += result
        return base
    
    def countAndSay(self, n): # more understandable version
    
        base = "1"  # Starting base case
        
        for i in range(n - 1):
            temp = base  # Current sequence
            base = ""  # Reset for the new sequence
            count = 1  # Initialize count for the first character
            
            for j in range(1, len(temp)):  # Start from 1 to compare with temp[0]
                if temp[j] == temp[j - 1]:  # If same as previous char
                    count += 1
                else:  # When characters change, append count and char
                    base += str(count) + temp[j - 1]
                    count = 1  # Reset count for the new character
                    
            # Add the last counted group
            base += str(count) + temp[-1]
        
        return base
    
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        def backtrack(remainder, combination, start):
            if remainder == 0:
                result.append(list(combination))
                return 
            elif remainder < 0:
                return
            
            for i in range(start, len(candidates)):
                combination.append(candidates[i])
                # number can be reused, we pass i not i+1
                backtrack(remainder - candidates[i], combination, i)
                # backtrack by removing the last added candidate
                combination.pop()
              
        backtrack(target, [], 0)
        return result
    
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res_perms = [[]]
        
        for num in nums:
            new_perm = []

            for perm in res_perms:
                for i in range(len(perm)+1):
                    new_perm.append(perm[:i] + [num] + perm[i:])

            res_perms = new_perm

        return res_perms

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        candidates.sort()
        def backtrack(remainder, combination, start):
            if remainder == 0:
                result.append(list(combination))
            elif remainder < 0:
                return
        
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                combination.append(candidates[i])
                backtrack(remainder - candidates[i], combination, i+1)
                combination.pop()

        backtrack(target, [], 0)
        return result
    
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        for i in range(len(matrix)):
            for j in range(i, len(matrix)):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        for i in range(len(matrix)):
            matrix[i] = reversed(matrix[i])
    
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        
        for r in range(9):
            for c in range(9):
                num = board[r][c]
                if num == ".":
                    continue
                
                if num in rows[r]:
                    return False
                rows[r].add(num)
                
                if num in cols[c]:
                    return False
                cols[c].add(num)
                
                box_index = (r // 3) * 3 + (c // 3)
                if num in boxes[box_index]:
                    return False
                boxes[box_index].add(num)
        
        return True

    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        results = collections.defaultdict(list)
        for string in strs:
            sorted_string = ''.join(sorted(string))
            results[sorted_string].append(string)
        return list(results.values())
    
    def lexicalOrder(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        nums = [str(i) for i in range(1,n+1)]
        sorted_nums = sorted(nums, key=str)
        sorted_nums_int = [int(i) for i in sorted_nums]
        return sorted_nums_int
    
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res_perms = [[]]
        
        for num in nums:
            new_perm = []

            for perm in res_perms:
                for i in range(len(perm)+1):
                    new_perm.append(perm[:i] + [num] + perm[i:])

            res_perms = new_perm
        
        res_perms.sort()
        return list(res_perms for res_perms, _ in itertools.groupby(res_perms))

    # TIME LIMIT EXCEEDED
    def isValidSudoku(self, board):
            rows = [set() for _ in range(9)]
            cols = [set() for _ in range(9)]
            boxes = [set() for _ in range(9)]

            for r in range(9):
                for c in range(9):
                    num = board[r][c]
                    if num == ".":
                        continue

                    if num in rows[r]:
                        return False
                    rows[r].add(num)

                    if num in cols[c]:
                        return False
                    cols[c].add(num)

                    box_index = (r // 3) * 3 + (c // 3)
                    if num in boxes[box_index]:
                        return False
                    boxes[box_index].add(num)

            return True
        
        def backtrack(board):
            for r in range(9):
                for c in range(9):
                    if board[r][c] == '.':
                        for num in map(str, range(1, 10)):
                            if isValidSudoku(board):
                                board[r][c] = num
                                if backtrack(board):
                                    return True
                                board[r][c] = '.'
                        return False
            print(board)
            return True
            
        backtrack(board)
    
    # MEMORY LIMIT EXCEEDED for large n
    def findKthNumber(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        nums = [str(i) for i in range(1,n+1)]
        sorted_nums = sorted(nums, key=str)
        sorted_nums_int = [int(i) for i in sorted_nums]
        return sorted_nums_int[k-1]
    
    # BETTER SOLUTION: PREFIX-BASED TREE TRAVERSAL (SIMILAR TO A TRIE)
    def findKthNumber2(self, n, k):
        def countSteps(n, curr, next):
            steps = 0
            while curr <= n:
                steps += min(n + 1, next) - curr
                curr *= 10
                next *= 10
            return steps
        
        curr = 1
        k -= 1  
        
        while k > 0:
            steps = countSteps(n, curr, curr + 1)
            if steps <= k:
                curr += 1
                k -= steps
            else:
                curr *= 10
                k -= 1
                
        return curr
    
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        
        prehead = ListNode(0, head)
        node_before_sublist = prehead
        while True:
            initial_starting_node = node_before_sublist.next
            initial_kth_node = self.get_kth_node(node_before_sublist, k)
            if initial_kth_node == None:
                break
            node_after_sublist = initial_kth_node.next
            
            
            prev_node = None
            current_node = node_before_sublist.next
            while current_node != node_after_sublist:
                next_node = current_node.next
                
                current_node.next = prev_node
                
                prev_node = current_node
                current_node = next_node
            
            node_before_sublist.next = initial_kth_node
            initial_starting_node.next = node_after_sublist
            
            node_before_sublist = initial_starting_node
        
        return prehead.next
    
    def get_kth_node(self, prev_node, k):
        current_node = prev_node
        while current_node and k > 0:
            current_node = current_node.next
            k -= 1
        
        return current_node

    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        res = str(int(num1) * int(num2))
        return res

    def multiply2(self, num1: str, num2: str) -> str:
        n1 = 0
        n2 = 0
        for i in num1:
            n1 = n1*10 + (ord(i) - 48)
        for i in num2:
            n2 = n2*10 + (ord(i) - 48)
        ans = str(n1*n2)
        return ans
    
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        MAX_INT = 2**31 - 1
        MIN_INT = -2**31
        
        if dividend == MIN_INT and divisor == -1:
            return MAX_INT
        
        sign = -1 if (dividend < 0) != (divisor < 0) else 1
        
        dividend, divisor = abs(dividend), abs(divisor)
        quotient = 0
        
        # subtract divisor from dividend, but do it efficiently
        while dividend >= divisor:
            temp_divisor, multiple = divisor, 1
            
            # efficiently subtract powers of two of the divisor
            while dividend >= (temp_divisor << 1):
                temp_divisor <<= 1
                multiple <<= 1
                
            dividend -= temp_divisor
            quotient += multiple
        
        result = sign * quotient
    
    def sublist_with_largest_sum(lst):
        max_sum = float('-inf')  # Initialize to negative infinity to handle negative numbers
        sublist = []
        
        for i in range(len(lst)):
            current_sum = 0
            for j in range(i, len(lst)):
                current_sum += lst[j]
                if current_sum > max_sum:
                    max_sum = current_sum
                    sublist = lst[i:j+1]  # Update sublist with current sublist
        
        return sublist, max_sum
    
    ## OR use Kandane's Algorithm: O(n) time
    def max_sub_array_sum(nums):
        current_sum = max_sum = nums[0] # Init both to the first element

        for num in nums[1:]:
            current_sum = max(num, current_sum + num) # Decide whether to add num or start a new subarray
            max_sum = max(max_sum, current_sum) # Update the maximum sum so far

        return max_sum
        
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        top = 0
        bottom = len(matrix) - 1
        left = 0
        right = len(matrix[0]) - 1
        spiral = []
        
        while (top <= bottom) and (left <= right):
            for i in range(left, right + 1):
                spiral.append(matrix[top][i])
            top += 1
            for i in range(top, bottom + 1):
                spiral.append(matrix[i][right])
            right -= 1
            
            if top <= bottom and left <= right:
                
                for i in range(right, left -1, -1):
                    spiral.append(matrix[bottom][i])
                bottom -= 1
                
                for i in range(bottom, top - 1, -1):
                    spiral.append(matrix[i][left])
                left += 1
        return spiral
    
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals:
            return []
        
        intervals.sort(key=lambda x: x[0])
        
        result = [intervals[0]]
        
        for i in range(1, len(intervals)):
            # Get the last interval in the result
            last_interval = result[-1]
            # If the current interval overlaps with the last interval, merge them
            if intervals[i][0] <= last_interval[1]:
                last_interval[1] = max(last_interval[1], intervals[i][1])
            else:
                # Add the current interval to the result
                result.append(intervals[i])
                
        return result
    
    def lengthOfLastWord(self, s):
        last_word = s.strip().split(" ")[-1]
        print("last word:", last_word)
        return len(last_word)
    
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n <= 0:
            return []
        
        matrix = [row[:] for row in [[0]*n]*n]
        row_st=0
        row_end=n-1
        
        col_st=0
        col_end=n-1
        current=1
        
        while(True):
            if current > n*n:
                break
            for c in range(col_st, col_end+1):
                matrix[row_st][c] = current
                current += 1
            row_st += 1
            for r in range(row_st, row_end+1):
                matrix[r][col_end] = current
                current += 1
            col_end-=1
            for c in range(col_end, col_st-1, -1):
                matrix[row_end][c] = current
                current+=1
            row_end-=1
            for r in range(row_end, row_st-1, -1):
                matrix[r][col_st] = current
                current+=1
            col_st+=1
        
        return matrix
    
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[1]*n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0

        obstacleGrid[0][0] = 1

        for i in range(1, m):
            obstacleGrid[i][0] = 1 if obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1 else 0

        for j in range(1, n):
            obstacleGrid[0][j] = 1 if obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1 else 0

        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    obstacleGrid[i][j] = 0
                else:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]

        return obstacleGrid[m-1][n-1]
    
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]
        
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        
        return grid[m-1][n-1]
    
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a_int = int(a, 2)
        b_int = int(b, 2)
        sum = a_int + b_int
        sum_new = "{0:08b}".format(sum).lstrip("0") if sum > 0 else "0"
        return sum_new
    
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        num = int(''.join(map(str, digits))) + 1
        num_list = [int(i) for i in str(num)]
        return num_list


### TESTING ###

def create_linked_list(lst):
    dummy_head = ListNode(0)
    current = dummy_head

    for value in lst:
        current.next = ListNode(value)
        current = current.next
    
    return dummy_head.next

def print_linked_list(node):
    values = []
    while node:
        values.append(str(node.val))
        node = node.next
    print(" -> ".join(values))

l1_values = [2, 4, 3]
l2_values = [5, 6, 4]

l1 = create_linked_list(l1_values)
l2 = create_linked_list(l2_values)

s = Solution()
print_linked_list(s.addTwoNumbers(l1, l2))    

s_p =  "babad"
print(s.longestPalindrome_2(s_p))

