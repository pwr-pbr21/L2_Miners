from tasks import *
import sys

print("Begin of Tasks")
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
        log("Begin of Task 1")
        task1()
        log("End of Task 1")
    elif(task == 2):
        log("Begin of Task 2")
        task2()
        log("End of Task 2")
    elif(task == 3):
        log("Begin of Task 3")
        task3()
        log("End of Task 3")
    elif(task == 4):
        log("Begin of Task 4")
        task4()
        log("End of Task 4")
    elif(task == 5):
        log("Begin of Task 5")
        task5()
        log("Begin of Task 5")
    else:
        raise ValueError("Invalid argument")

print("End of Task")