
device=1
folder=v0904
result_json=./score/result_slot_$folder.json

for seed in {1..3}; do
    for fold in {0..0}; do
        # MatSciBERT
        
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_none/train.txt --name matsci_eq --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_binomial/train.txt --name matsci_bin --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_poisson/train.txt --name matsci_poi --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_geometric/train.txt --name matsci_geo --device $device --result_json $result_json
        
        
        # SciBERT
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_none/train.txt --model allenai/scibert_scivocab_uncased --name sci_eq --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_binomial/train.txt --model allenai/scibert_scivocab_uncased --name sci_bin --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_poisson/train.txt --model allenai/scibert_scivocab_uncased --name sci_poi --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_geometric/train.txt --model allenai/scibert_scivocab_uncased --name sci_geo --device $device --result_json $result_json
        
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_GPT35_base/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug --device $device --result_json $result_json
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_llmda/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug2 --device $device --result_json $result_json

        # BERT
        
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_none/train.txt --name bert_eq --model bert-base-uncased --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_binomial/train.txt --name bert_bin --model bert-base-uncased --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_poisson/train.txt --name bert_poi --model bert-base-uncased --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_geometric/train.txt --name bert_geo --model bert-base-uncased --device $device --result_json $result_json
        
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_GPT35_base/train.txt --name bert_gptaug --model bert-base-uncased --device $device --result_json $result_json
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_llmda/train.txt --name bert_gptaug2 --model bert-base-uncased --device $device --result_json $result_json
        
        # MatBERT
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_none/train.txt --model ../data/model/matbert/ --name mat_eq  --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_binomial/train.txt --model ../data/model/matbert/ --name mat_bin  --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_poisson/train.txt --model ../data/model/matbert/ --name mat_poi  --device $device --result_json $result_json
        python run_SOFC_NER.py -fp16 -slot -detail --seed $seed --fold $fold --augment ../augmentation/$folder/Slot_geometric/train.txt --model ../data/model/matbert/ --name mat_geo  --device $device --result_json $result_json
        
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_GPT35_base/train.txt --model ../data/model/matbert/ --name mat_gptaug  --device $device --result_json $result_json
        #python run_SOFC_NER.py -fp16 -crf -slot -detail --seed $seed --fold $fold --augment ../Slot_llmda/train.txt --model ../data/model/matbert/ --name mat_gptaug2  --device $device --result_json $result_json
    done
done