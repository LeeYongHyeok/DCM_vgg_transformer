#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Prepare librispeech dataset

train_dir=spm_text_char

#audio_dir=${1%/}	#/home/nas/[DB]_LIPREADING/LRS_con_wav
#video_dir=${2%/}	#/home/nas/[DB]_LIPREADING/LRS_con_mp4_feat
#out_dir=${3%/}		#/home/nas/[DB]_LIPREADING/LRS

audio_dir="/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/raw_data/audio/LRS_con_wav"
video_dir="/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/raw_data/video/LRS_con_mp4_feat"
out_dir="/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/preprocessed_data/character/sentence_pretrain_trainval"

fairseq_root="/home/nas/user/yong/fairseq"

nbpe=5000
bpemode=char

#rm -rf $out_dir

if [ ! -d "$fairseq_root" ]; then
    echo "$0: Please set correct fairseq_root"
    exit 1
fi
if [ ! -d "$audio_dir" ]; then
	echo "audio_dir is not exist"
	exit 1
fi
if [ ! -d "$video_dir" ]; then
	echo "video_dir is not exist"
	exit 1
fi
if [ ! -d "$out_dir" ]; then
	mkdir -p ${out_dir}
fi

cd ${out_dir} || exit

echo "audio text file merge"

#for audio_folder in pretrain_subvid_enc_0.5 test trainval; do
#	mkdir -p ${out_dir}/${audio_folder}
#	find ${audio_dir}/${audio_folder}/ -name "*.txt" -exec head -n 1 {} \; >> ${out_dir}/$audio_folder/text
#done
echo "textfile finish"
cat ${out_dir}/pretrain/text ${out_dir}/trainval/text ${out_dir}/test/text > ${out_dir}/${train_dir}/text

dict=data/lang_char/${train_dir}_${bpemode}${nbpe}_units.txt
encoded=data/lang_char/${train_dir}_${bpemode}${nbpe}_encoded.txt
fairseq_dict=data/lang_char/${train_dir}_${bpemode}${nbpe}_fairseq_dict.txt


bpemodel=data/lang_char/${train_dir}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
echo "Dictionary preparation"
mkdir -p data/lang_char/
echo "<unk> 3" > ${dict}
echo "</s> 2" >> ${dict}
echo "<pad> 1" >> ${dict}

cut -f 2- -d" " ${out_dir}/${train_dir}/text > data/lang_char/input.txt
echo "spm model training start"
#spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --unk_id=3 --eos_id=2 --pad_id=1 --bos_id=-1 --character_coverage=1
spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt > ${encoded}
cat ${encoded} | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+3}' >> ${dict}
cat ${encoded} | tr ' ' '\n' | sort | uniq -c | awk '{print $2 " " $1}' > ${fairseq_dict}
wc -l ${dict}

echo "Prepare audio & video jsons"
#for part in trainval test pretrain; do
for part in test trainval pretrain; do
	echo "${part} start"
    python ${fairseq_root}/examples/audio_visual_speech_recognition/datasets/asr_prep_json_LRS.py --audio-dirs ${audio_dir}/${part} --video-dirs ${video_dir}/${part} --labels ${out_dir}/${part}/text --spm-model ${bpemodel}.model --audio-format wav --dictionary ${fairseq_dict} --output ${part}.json
done

cp ${fairseq_dict} ./dict.txt
cp ${bpemodel}.model ./spm.model
