#! /bin/bash
#author:Shelsin
#use read in the script

for ((i=3;i<=13;i++))
do
  echo "python main.py --content_dir datasets/content --content_name $i --style_dir datasets/content --style_name $i --model_name MST --step 11"
  eval "python main.py --content_dir datasets/content --content_name $i --style_dir datasets/content --style_name $i --model_name MST --step 11"
done
for ((i=2;i<=13;i++))
do
  echo "python main.py --content_dir datasets/content --content_name $i --style_dir datasets/style --style_name $i --model_name MST --step 11"
  eval "python main.py --content_dir datasets/content --content_name $i --style_dir datasets/style --style_name $i --model_name MST --step 11"
done
