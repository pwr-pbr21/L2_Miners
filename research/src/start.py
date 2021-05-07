from tasks import *
import sys

if len(sys.argv) == 1:
    task1()
    task2()
    task3()
    task4()
    task5()
else:
    try:
        task = int(sys.argv[1])
    except:
        raise ValueError("Invalid argument")
        
    if(task == 1):
        task1()
    elif(task == 2):
        task2()
    elif(task == 3):
        task3()
    elif(task == 4):
        task4()
    elif(task == 5):
        task5()
    else:
        raise ValueError("Invalid argument")

print("End of Task")