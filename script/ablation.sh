device=1
seed=123
folder=v0809

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/0_baseline --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_Bert  --model bert-base-uncased --mlm_p .7 --KG_p 0 --resample_length none --round 3

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/1_lm2matsci --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_MatSci --mlm_p .7 --KG_p 0 --resample_length none
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/2_matkg --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_Bert --model bert-base-uncased --resample_length none
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/3_length --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_Bert --model bert-base-uncased --mlm_p .7 --KG_p 0 

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/12 --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_MatSci --resample_length none 
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/23 --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_Bert --model bert-base-uncased
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/13 --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_MatSci --mlm_p .7 --KG_p 0 

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_MatSci/123 --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_MatSci

python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/0_baseline --data sofc --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_Bert  --model bert-base-uncased --mlm_p .7 --KG_p 0 --resample_length none --round 3

python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/1_lm2matsci --data sofc --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_MatSci --mlm_p .7 --KG_p 0 --resample_length none
python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/2_matkg --data sofc --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_Bert --model bert-base-uncased --resample_length none
python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/3_length --data sofc --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_Bert --model bert-base-uncased --mlm_p .7 --KG_p 0 

python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/12 --data sofc --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_MatSci --resample_length none 
python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/23 --data sofc --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_Bert --model bert-base-uncased
python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/13 --data sofc --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_MatSci --mlm_p .7 --KG_p 0 

python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Sofc_MatSci/123 --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Sofc_MatSci


result_json=./score/result_sofc_ablation_$folder.json

for seed in {1..3}; do
    for fold in {0..0}; do
        # MatSciBERT
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/0_baseline/train.txt --name matsci_0_baseline --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/1_lm2matsci/train.txt --name matsci_1_lm2matsci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/2_matkg/train.txt --name matsci_2_matKG --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/3_length/train.txt --name matsci_3_length --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/12/train.txt --name matsci_12_lmNkg --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/23/train.txt --name matsci_23_kgNlen --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/13/train.txt --name matsci_13_lmNlen --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Sofc_MatSci/123/train.txt --name matsci_123_final --device $device --result_json $result_json

        # MatBERT
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/0_baseline/train.txt --name mat_0_baseline --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/1_lm2matsci/train.txt --name mat_1_lm2matsci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/2_matkg/train.txt --name mat_2_matKG --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/3_length/train.txt --name mat_3_length --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/12/train.txt --name mat_12_lmNkg --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/23/train.txt --name mat_23_kgNlen --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/13/train.txt --name mat_13_lmNlen --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model ../data/model/matbert/ --augment ../augmentation/$folder/Sofc_MatSci/123/train.txt --name mat_123_final --device $device --result_json $result_json

        # SciBERT
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/0_baseline/train.txt --name sci_0_baseline --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/1_lm2matsci/train.txt --name sci_1_lm2matsci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/2_matkg/train.txt --name sci_2_matKG --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/3_length/train.txt --name sci_3_length --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/12/train.txt --name sci_12_lmNkg --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/23/train.txt --name sci_23_kgNlen --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/13/train.txt --name sci_13_lmNlen --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/Sofc_MatSci/123/train.txt --name sci_123_final --device $device --result_json $result_json

        # BERT
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/0_baseline/train.txt --name bert_0_baseline --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/1_lm2matsci/train.txt --name bert_1_lm2matsci --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/2_matkg/train.txt --name bert_2_matKG --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/3_length/train.txt --name bert_3_length --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/12/train.txt --name bert_12_lmNkg --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/23/train.txt --name bert_23_kgNlen --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/13/train.txt --name bert_13_lmNlen --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -detail --seed $seed --fold $fold --model bert-base-uncased --augment ../augmentation/$folder/Sofc_MatSci/123/train.txt --name bert_123_final --device $device --result_json $result_json
        done
done
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_MatSci --data sofc --merge_file data/task/sofc_process/train.txt #--ner_src data/model/Sofc_MatSci
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_MatSci --data matscholar --merge_file data/task/matscholar/train.txt 
