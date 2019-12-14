#!/usr/bin/env sh

for input in $(ls '../dataset/recordings/stage-2' | egrep '\.mp3$')
   do 
      sox "../dataset/recordings/stage-2/$input" "../dataset/recordings/stage-2/converted/$input" compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1 lowpass 3000
      echo "noise gate and lowpass applied in $input"
   done 


#for input in $(ls '../dataset/recordings/stage-2' | egrep '\.mp3$')
   #do 
      #sox "../dataset/recordings/stage-2/$input" "../dataset/recordings/stage-2/converted/$input" lowpass 3000 compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1
      #echo "noise gate and lowpass applied in $input"
   #done

