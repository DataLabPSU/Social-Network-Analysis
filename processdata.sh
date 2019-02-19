#!/bin/sh  
while true  
do 
echo "calling processdata"
wget http://127.0.0.1:8000/processdata -O /dev/null
sleep 60
done