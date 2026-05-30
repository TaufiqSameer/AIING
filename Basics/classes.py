class Product:
    # public variables
    _name : str
    _age : int
    # protected -> underscore
    def __init__(self : None,name : str,age : int):
        self._name = name
        self._age = age
        
    
    def getName(self):
        return self._name

    def getAge(self):
        return self._age
    

class item(Product):
    itemno = 0;
    
    def __init__(self,name,age,itemno):
        super().__init__(name,age);
        self.itemno = itemno
        
    
        
p1 = Product("sameer",19)

print(p1.getName())
print(p1.getAge())
print(p1._name)
p2 = item("sameer",19,173);
print(p2.getName())
print(p2.itemno)
print(dir(p1))
        
    