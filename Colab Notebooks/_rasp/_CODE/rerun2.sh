#!/bin/bash
echo `date` > out
sleep 10
echo `date` >> out
killall python3
python3 client2.py >>out  &
