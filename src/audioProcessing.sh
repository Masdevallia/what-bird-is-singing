#!/usr/bin/env sh

path=${1}

# Applying low pass filter + noise gate:

for input in $(ls $path | egrep '\.mp3$')
   do 
      sox "$path/$input" "$path/converted/$input" lowpass 3000 compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1
      echo "lowpass and noise gate applied in $input"
   done
