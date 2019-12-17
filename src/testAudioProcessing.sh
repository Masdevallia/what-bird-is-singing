
filename=${1}
path='./application/uploaded'

# lowpass + noise gate:

for input in $(ls $path | egrep '\.mp3$')
   do
      if [ $input == $filename ]; then
         sox "$path/$input" "$path/converted/$input" lowpass 3000 compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1
         echo "lowpass and noise gate applied in $input"
      fi
   done

