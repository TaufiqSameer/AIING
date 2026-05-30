def mul(x: int) -> int:
    return x * 2

nums = [10, 20, 30]

list2 = list(map(mul, nums))

print(list2)

##lambda - function with no name -> i.e, Anonmyous function

def addition(a,b) :
    return a + b;

print(addition(10,20));

add = (lambda a=10,b=10 : a + b)

print(add())

def getEven(n : int) -> bool :
    if n % 2 == 0 :
        return True
    else:
        return False

lst = [i for i in range(0,10)]

print(lst)

lst2 = list(filter(getEven,lst));
print(lst2);

## List comprehension

def lstsqr(lst):
    lst1 = []
    for i in lst: 
        i = i * 2
        lst1.append(i)
    return lst1

lst3 = lstsqr(lst2)
print(lst3)

lst4 = [z for z in range (1,20) if z % 2 == 0]

print(lst4)

# name = input("Enter your name")
# lmao = "this is {} ...".format(name);

# print(lmao)

#python iterables and iterators

it = iter(lst4)

for i in it:
    print(i)
