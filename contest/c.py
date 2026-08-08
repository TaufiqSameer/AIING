t = int(input())

for _ in range(t):
    n = int(input())
    a = list(input())
    b = list(input())
    
    e1 = 0
    e2 = 0
    e3 = 0
    e4 = 0

    for i in range(n):
        if i % 2 == 0:
            e1 += (a[i] == '1')
            e3 += (b[i] == '1')
        else:
            e2 += (a[i] == '1')
            e4 += (b[i] == '1')

    if e1 == e3 and e2 == e4:
        print("Yes")
    else:
        print("No")
    
    # if n == 1:
    #     if a[0] == b[0]:
    #         print("Yes")
    #         continue
    #     else:
    #         print("No")
    #         continue
    # can = False
    # for i in range(n-2):
    #     c = a[:]
    #     if c[i] == '0' and c[i+1] == '0' and c[i+2] == '1':
    #         c[i] = '1'
    #         c[i+1] = '0'
    #         c[i+2] = '0'
    #         if c == b:
    #             can = True
    #             break
    #     c = a[:]
    #     if c[i] == '1' and c[i+1] == '0' and c[i+2] == '0':
    #         c[i] = '0'
    #         c[i+1] = '0'
    #         c[i+2] = '1'
    #         if c == b:
    #             can = True
    #             break
            
    # for i in range(n-2):
    #     c = a[:]
    #     if c[i] == '1' and c[i+1] == '1' and c[i+2] == '0':
    #         c[i] = '0'
    #         c[i+1] = '1'
    #         c[i+2] = '1'
    #         if c == b:
    #             can = True
    #             break
    #     c = a[:]
    #     if c[i] == '0' and c[i+1] == '1' and c[i+2] == '1':
    #         c[i] = '1'
    #         c[i+1] = '1'
    #         c[i+2] = '0'
    #         if c == b:
    #             can = True
    #             break
        
    # if can:
    #     print("Yes")
    # else:
    #     print("No")

