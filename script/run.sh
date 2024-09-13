start=`date +%s`

#bash run_SOFC_slot.sh
#bash run_SOFC.sh



device=1
seed=123
folder=v0809

python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt -use_crf --ner_epochs 40
python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_MatSci --data sofc --merge_file data/task/sofc_process/train.txt -use_crf
python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_MatSci --data matscholar --merge_file data/task/matscholar/train.txt -use_crf

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_Sci --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --model allenai/scibert_scivocab_uncased -use_crf --ner_epochs 40
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_Sci --data sofc --merge_file data/task/sofc_process/train.txt --model allenai/scibert_scivocab_uncased -use_crf
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_Sci --data matscholar --merge_file data/task/matscholar/train.txt --model allenai/scibert_scivocab_uncased -use_crf

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_Mat --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --model ../data/model/matbert/ -use_crf --ner_epochs 40
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_Mat --data sofc --merge_file data/task/sofc_process/train.txt --model ../data/model/matbert/ -use_crf
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_Mat --data matscholar --merge_file data/task/matscholar/train.txt --model ../data/model/matbert/ -use_crf

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_Bert --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --model bert-base-uncased -use_crf --ner_epochs 40
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_Bert --data sofc --merge_file data/task/sofc_process/train.txt --model bert-base-uncased -use_crf
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_Bert --data matscholar --merge_file data/task/matscholar/train.txt --model bert-base-uncased -use_crf



#bash run_SOFC_slot.sh
#bash run_SOFC.sh
#bash run_matscholar.sh

### END ###

result_json=./score/result_mine_$folder.json

for seed in {1..3}; do
    for fold in {0..0}; do
        # matscholar
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MatSci/train.txt --name matsch_matsci --device $device --result_json $result_json
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MatSci/train.txt --model bert-base-uncased --name matsch_bert --device $device --result_json $result_json
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MatSci/train.txt --model allenai/scibert_scivocab_uncased --name matsch_sci --device $device --result_json $result_json
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MatSci/train.txt --model ../data/model/matbert/ --name matsch_mat --device $device --result_json $result_json
        # SOFC
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MatSci/train.txt --name sofc_matsci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MatSci/train.txt --model bert-base-uncased --name sofc_bert --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MatSci/train.txt --model allenai/scibert_scivocab_uncased --name sofc_sci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MatSci/train.txt --model ../data/model/matbert/ --name sofc_mat --device $device --result_json $result_json
        # Slot
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --name slot_matsci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --model bert-base-uncased --name slot_bert --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --model allenai/scibert_scivocab_uncased --name slot_sci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --model ../data/model/matbert/ --name slot_mat --device $device --result_json $result_json
        
    done
done