num1 = float(input("Enter the first number"))
num2 = float(input("Enter the first number"))

ch : chr = input("Enter the operand")

num : int;

match ch:
    case '+':
        num = num1 + num2
    case '-' :
        num = num1 - num2
    case '/':
        num = num1 / num2
    case '%':
        num = num1 % num2
    case _:
        print("Enter an valid operation")


print(f"{num1} {ch} {num2} = {num}")    
