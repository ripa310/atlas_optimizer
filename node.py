from typing import List

class Node:
    def __init__(self, id , name, source:List[str], target:List[str]):
        self.id = id
        self.name = name
        self.source = source
        if(isinstance(target, int)):
            print("Target int : ", target)
        self.target = target
    
    def __repr__(self) -> str:
        return f"node id = {self.id}, node name = {self.name}"
