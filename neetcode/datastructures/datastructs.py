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
