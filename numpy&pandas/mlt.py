'''
matpotlib is plotting library and seaborn
provides object orient api for embeeding plots into applications using general purpose GUI toolkits like Tkinter, wxpython
'''
import matplotlib.pyplot as plt;
import numpy as np;
x = np.arange(0,10);
y = np.arange(11,21);

print(x,y)

plt.scatter(x,y,c='r');
plt.xlabel("number1")
plt.ylabel("number2")
plt.title("my graph")
plt.show()
y = x * x
print(y)
plt.plot(x,y,'ro--')
plt.show()

plt.subplot(2,2,1);
plt.plot(x,y,'r');
plt.show()
plt.subplot(2,2,2);
plt.plot(x,y,'g')
plt.show()

x = np.arange(0,4*np.pi,0.1);
print(x)
y = np.sin(x)
print(y)
plt.title("sin graph");

plt.plot(x,y);
plt.show();

x=[1,2,3]
y = [10,20,30]
plt.bar(x,y)
plt.show();

a=  np.arange(10,20,0.1);

plt.hist(a);
plt.show();


