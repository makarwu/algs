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
        print("dummy_head: ", dummy_head)
        current = dummy_head  # pointer to base list
        carry = 0 # carry if addition result > 9

        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            total = val1 + val2 + carry
            carry = total // 10
            current.next = ListNode(total % 10)
            print("current.next: ", current.next)
            current = current.next
            print("current: ", current)

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy_head.next

### TESTING ###
s = Solution()
l1 = None
l2 = None

print(s.addTwoNumbers(l1, l2))    
