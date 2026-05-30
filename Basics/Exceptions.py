x = int(input("Enter the number:"));

import os;
try:
    y = 10 / x
    print(y)
    with open("input.txt") as f1:
        print(f1.read());
except ZeroDivisionError:
    print("x is zero")
except FileNotFoundError as e:
    print(f"File not found for file {__file__}");
