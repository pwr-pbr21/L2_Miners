#!/bin/sh

python -u -W ignore github.py 2>&1 | tee ./out/log.txt
