#!/bin/sh

python -u -W ignore start.py $1 2>&1 | tee /outputData/task$1.log
