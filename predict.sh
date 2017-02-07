#!/usr/bin/env bash

#############################################
DIR=.

# and, make sure ffmpeg is installed
FFMPEGBIN=ffmpeg
#############################################

for f in ${DIR}/*.mov; do
  dir=${f::-4}
  echo -----
  echo Extracting frames from ${f} into ${dir}...
  if [[ ! -d ${dir} ]]; then
    echo Creating directory=${dir}
    mkdir -p ${dir}
  fi
  #segmentation
  ${FFMPEGBIN} -i ${f} ${dir}/image_%4d.jpg
  
  #forward
  python gesture.py

  #synthesis
  cd ${dir}
  ${FFMPEGBIN} -r 20 -i ./image_%04d.jpg -vcodec mpeg4 ./result.mov
done

echo -------------------------------------------
echo Done!
