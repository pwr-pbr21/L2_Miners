#!/bin/sh

python -W ignore start.py 1 > /outputData/task1.log 2>&1 &
python -W ignore start.py 2 > /outputData/task2.log 2>&1 &
python -W ignore start.py 3 > /outputData/task3.log 2>&1 &
python -W ignore start.py 4 > /outputData/task4.log 2>&1 &

wait