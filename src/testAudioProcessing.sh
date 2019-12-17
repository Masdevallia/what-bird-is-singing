
filename=${1}

# lowpass + noise gate:
for input in $(ls './dataset/test' | egrep '\.mp3$')
   do
      if [ $input == $filename ]; then
         sox "./dataset/test/$input" "./dataset/test/converted/lp/$input" lowpass 3000
         echo "lowpass applied in $input"
      fi
   done

for input in $(ls './dataset/test/converted/lp' | egrep '\.mp3$')
   do 
      sox "./dataset/test/converted/lp/$input" "./dataset/test/converted/lp_ng/$input" compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1
      echo "lowpass and noise gate applied in $input"
   done
