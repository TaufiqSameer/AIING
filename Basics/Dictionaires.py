from collections import Counter
d1 = { "name" : "sameer","age" : 10}
print(d1["name"]);
print(d1.values())
print(type(d1))
for i in d1.keys():
    print(d1[i])
l1 = [10,10,10,20,20]
s = d1.items()
print(s)
l2 = Counter(l1)
print(l2)
print(d1.get("name"))

# word = "sameer"
# guess : str = ""
# c : int = 0 
# while word != guess:
#     guess = input("Enter the word")
#     c += 1

print(max(l2))
# print(f"You took {c} tries")

try:
    num = int(input("Enter a number"))
except ValueError:
    print("invalid")
except KeyboardInterrupt: 
    print("i")
except:
    print("F")
    
    
    



