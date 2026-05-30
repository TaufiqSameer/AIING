l1 = ["samer","ubed","humaid"]
l1[0] = "sameer"
print(l1)
print(l1[::-1])
print(l1[2:])

l2 = l1.copy()
print(all(l1) == 10)
l2.append(100)

print(l2)

#Tuples"""
#Typling
# """
(a,b,c) = (1,2,3)

points3d = (1,2,3)
points4d = (2,2,3)

print(a,b,c)

print(points3d.__le__(points4d))

print(max(1,2,3))

l1 = [10,20,30,40]

print(max(l1))

