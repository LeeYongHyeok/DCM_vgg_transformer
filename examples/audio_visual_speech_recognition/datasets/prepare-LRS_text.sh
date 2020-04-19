#!/usr/bin/env bash

audio_dir="/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/raw_data/audio"
video_dir="/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/raw_data/video"

if [ ! -d "$audio_dir" ]; then
	echo "audio_dir is not exist"
	exit 1
fi
if [ ! -d "$video_dir" ]; then
	echo "video_dir is not exist"
	exit 1
fi

## txt change text: -> utt_folder/number of file
#for folder in pretrain test trainval; do
for folder in pretrain_subvid_enc_0.5; do
	ls $audio_dir/$folder | while read line
	do
		ls -I*wav ${audio_dir}/${folder}/${line} | while read txtfile 
		do
			sed -i "s/Text\:/${folder}\/${line}\/${txtfile}/g" ${audio_dir}/${folder}/${line}/$txtfile
			sed -i "s/\.txt//g" ${audio_dir}/${folder}/${line}/$txtfile
		done
	done
done


