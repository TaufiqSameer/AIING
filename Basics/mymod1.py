def add(a : int , b : int) -> int :
    return a + b

def sub(a : int , b : int) -> int :
    return a -b 

def read_File(file : str) -> str :
    with open(file,"r") as f2:
        return f2.read();
