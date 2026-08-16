from pywebio.input import *
from pywebio.output import *
def bmicalculator():
    height = input("Enter the height(cms)",type=FLOAT)
    weight = input("Enter the weight(kg)",type=FLOAT)
    
    bmi = weight/(height/100)**2;
    
    bmi_output = [(16,"Serverly underweight"),(18.5,"Underweight"),(25,"Normal"),(30,"OVERWEIGht"),(35,"Moderately obese"),(float('inf'),'Serverly obese')]
    
    for t1,t2 in bmi_output:
        if bmi <= t1:
            put_text(f'BMI Index : {bmi:.1f} and according to bmi you are : {t2}')
            break

if __name__ == "__main__":
    bmicalculator()