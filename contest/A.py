t = int(input())

for _ in range(t):
    a, b, c = map(int, input().split())
    co = 0

    while a != b and b != c and a != c:
        co += 1

        if a > b and a > c:
            a -= 1
        elif b > a and b > c:
            b -= 1
        else:
            c -= 1


        if a < b and a < c:
            a += 1
        elif b < a and b < c:
            b += 1
        else:
            c += 1

    print(co)