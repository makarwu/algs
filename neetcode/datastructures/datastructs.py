### (1) DYNAMIC ARRAY ###

class DynamicArray:
    
    def __init__(self, capacity: int):
        self.arr = [''] * capacity
        self.capacity = capacity
        self.size = 0

    def get(self, i: int) -> int:
        return self.arr[i]

    def set(self, i: int, n: int) -> None:
        self.arr[i] = n

    def pushback(self, n: int) -> None:
        if self.size == self.capacity:
            self.resize()
        self.arr[self.size] = n
        self.size += 1

    def popback(self) -> int:
        if self.size > 0:
            last = self.arr[self.size - 1]
            self.size -= 1
            return last
        return None

    def resize(self) -> None:
        self.capacity *= 2
        new_arr = [''] * self.capacity
        for i in range(self.size):
            new_arr[i] = self.arr[i]

        self.arr = new_arr

    def getSize(self) -> int:
        return self.size
        
    def getCapacity(self) -> int:
        return self.capacity

### (2) Linked-List ###

class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None

    def get(self, index: int) -> int:
        current = self.head
        i = 0
        while current:
            if i == index:
                return current.value
            current = current.next
            i+=1
        return -1

    def insertHead(self, val: int) -> None:
        new_node = Node(val)
        new_node.next = self.head
        self.head = new_node

    def insertTail(self, val: int) -> None:
        new_node = Node(val)
        current = self.head
        if not self.head:
            self.head = new_node
            return
            
        while current.next:
            current = current.next
        current.next = new_node
        
    def remove(self, index: int) -> bool:
        if not self.head:
            return False
        if index == 0:
            self.head = self.head.next
            return True

        current = self.head
        i = 0
        while current and current.next:
            if i == index-1:
                current.next = current.next.next
                return True
            current = current.next
            i += 1

        return False


    def getValues(self) -> List[int]:
        current = self.head
        values = []
        while current:
            val = current.value
            values.append(val)
            current = current.next
        return values
