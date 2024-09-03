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

