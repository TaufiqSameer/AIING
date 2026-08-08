t = int(input())

for _ in range(t):
    n = int(input())
    s = input()
    gr = 1
    for i in range(1, n):
        if s[i] != s[i - 1]:
            gr += 1
    ans = gr
    for i in range(1, n - 1):
        cur = gr
        if s[i] != s[i - 1]:
            cur -= 1
        if s[i] != s[i + 1]:
            cur -= 1

        if s[i - 1] != s[i + 1]:
            cur += 1
        ans = min(ans, cur)
    print(ans)