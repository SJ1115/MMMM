
device=1
folder=v0904
result_json=./score/result_slot_$folder.json

for seed in {1..3}; do
    for fold in {0..0}; do
        # MatSciBERT
        python run_SOFC_NER.py -fp16 -slot -crf -detail --seed $seed --fold $fold --name matsci_base --device $device --result_json $result_json
        
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Bert/train.txt --name matsci_bertaug --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Sci/train.txt --name matsci_sciaug --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Mat/train.txt --name matsci_mataug --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --name matsci_matsciaug --device $device --result_json $result_json
        
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_GPT35_base/train.txt --name matsci_gptaug --device $device --result_json $result_json
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_llmda/train.txt --name matsci_gptaug2 --device $device --result_json $result_json
        
        # SciBERT
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --model allenai/scibert_scivocab_uncased --name sci_matsciaug --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Mat/train.txt --model allenai/scibert_scivocab_uncased --name sci_mataug --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Sci/train.txt --model allenai/scibert_scivocab_uncased --name sci_sciaug --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Bert/train.txt --model allenai/scibert_scivocab_uncased --name sci_bertaug --device $device --result_json $result_json
        
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --name sci_base --model allenai/scibert_scivocab_uncased --device $device --result_json $result_json
        
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_GPT35_base/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug --device $device --result_json $result_json
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_llmda/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug2 --device $device --result_json $result_json

        # BERT
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --name bert_base --model bert-base-uncased --device $device --result_json $result_json
        
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Bert/train.txt --name bert_bertaug --model bert-base-uncased --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Sci/train.txt --name bert_sciaug --model bert-base-uncased --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Mat/train.txt --name bert_mataug --model bert-base-uncased --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --name bert_matsciaug --model bert-base-uncased --device $device --result_json $result_json
        
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_GPT35_base/train.txt --name bert_gptaug --model bert-base-uncased --device $device --result_json $result_json
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_llmda/train.txt --name bert_gptaug2 --model bert-base-uncased --device $device --result_json $result_json
        
        # MatBERT
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --name matbert_base --model ../data/model/matbert/ --device $device --result_json $result_json

        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Bert/train.txt --model ../data/model/matbert/ --name mat_bertaug  --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Sci/train.txt --model ../data/model/matbert/ --name mat_sciaug  --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_Mat/train.txt --model ../data/model/matbert/ --name mat_mataug  --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_MatSci/train.txt --model ../data/model/matbert/ --name mat_matsciaug  --device $device --result_json $result_json
        
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_GPT35_base/train.txt --model ../data/model/matbert/ --name mat_gptaug  --device $device --result_json $result_json
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_llmda/train.txt --model ../data/model/matbert/ --name mat_gptaug2  --device $device --result_json $result_json
    done
done