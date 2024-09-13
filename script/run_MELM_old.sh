start=`date +%s`

#bash run_SOFC_slot.sh
#bash run_SOFC.sh



device=1
seed=123
folder=v0906


#python generate_MELM_old.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MELM --model FacebookAI/roberta-large --mlm_batch 2 --ner_batch 2 --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt -use_crf --ner_epochs 40
#python generate_MELM_old.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_MELM --model FacebookAI/roberta-large --mlm_batch 2 --ner_batch 2 --data sofc --merge_file data/task/sofc_process/train.txt -use_crf
#python generate_MELM_old.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_MELM --model FacebookAI/roberta-large --mlm_batch 2 --data matscholar --merge_file data/task/matscholar/train.txt -use_crf

python generate_MELM_old.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MELM --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt -use_crf --ner_epochs 40
python generate_MELM_old.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_MELM --data sofc --merge_file data/task/sofc_process/train.txt -use_crf
python generate_MELM_old.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_MELM --data matscholar --merge_file data/task/matscholar/train.txt -use_crf

result_json=./score/result_MELM_mat_$folder.json

for seed in {1..3}; do
    for fold in {0..0}; do
        # matscholar
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MELM/train.txt --name matsch_matsci_L --device $device --result_json $result_json
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MELM/train.txt --model bert-base-uncased --name matsch_bert_L --device $device --result_json $result_json
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MELM/train.txt --model allenai/scibert_scivocab_uncased --name matsch_sci_L --device $device --result_json $result_json
        python run_matscholar_NER.py -detail -crf -fp16 --seed $seed --augment ../augmentation/$folder/matsch_MELM/train.txt --model ../data/model/matbert/ --name matsch_mat_L --device $device --result_json $result_json
        # SOFC
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MELM/train.txt --name sofc_matsci_L --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MELM/train.txt --model bert-base-uncased --name sofc_bert_L --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MELM/train.txt --model allenai/scibert_scivocab_uncased --name sofc_sci_L --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/SOFC_MELM/train.txt --model ../data/model/matbert/ --name sofc_mat_L --device $device --result_json $result_json
        # Slot
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MELM/train.txt --name slot_matsci_L --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MELM/train.txt --model bert-base-uncased --name slot_bert_L --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MELM/train.txt --model allenai/scibert_scivocab_uncased --name slot_sci_L --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MELM/train.txt --model ../data/model/matbert/ --name slot_mat_L --device $device --result_json $result_json
        
    done
done