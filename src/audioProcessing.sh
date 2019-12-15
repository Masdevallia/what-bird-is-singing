#!/usr/bin/env sh

# converted
# for input in $(ls '../dataset/recordings/stage-2' | egrep '\.mp3$')
   # do 
      # sox "../dataset/recordings/stage-2/$input" "../dataset/recordings/stage-2/converted/$input" compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1 lowpass 3000
      # echo "noise gate and lowpass applied in $input"
   # done 

# converted2:
for input in $(ls '../dataset/recordings/stage-1' | egrep '\.mp3$')
   do 
      sox "../dataset/recordings/stage-1/$input" "../dataset/recordings/stage-1/converted2/$input" lowpass 3000 compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1
      echo "lowpass and noise gate applied in $input"
   done

