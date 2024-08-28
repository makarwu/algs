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
        max_length = 0 # result for the max length found
        start = 0 # start index of current sliding window
        for i, char in enumerate(s):
            # If the character is found in the dictionary and its index is within the current window
            if char in char_map and char_map[char] >= start:
                # move sliding window to the right
                start = char_map[char] + 1

            # Update the last position of the current character
            char_map[char] = i
            # calc max length
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
