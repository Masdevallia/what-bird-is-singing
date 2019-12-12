#!/usr/bin/env sh

for input in $(ls '../dataset/recordings/stage-1' | egrep '\.mp3$')
   do 
      sox "../dataset/recordings/stage-1/$input" "../dataset/recordings/stage-1/converted/$input" compand .1,.2 -inf,-50.1,-inf,-50,-50 0 -90 .1 lowpass 3000
      echo "noise gate and lowpass applied in $input"
   done 


# Añadir gain 10 para subir el volumen?
# Subir frecuencia en el lowpass?
# También podría añadir un highpass...
