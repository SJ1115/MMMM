
device=1
folder=v0731
result_json=./score/result_sofc_$folder.json


for seed in {1..3}; do
    # MatSciBERT
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_MatSci/train.txt --name matsci_matsciaug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Mat/train.txt --name matsci_mataug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Sci/train.txt --name matsci_sciaug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Bert/train.txt --name matsci_bertaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --name matsci_gptaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --name matsci_gptaug2 --device $device --result_json $result_json
    
    # SciBERT
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_MatSci/train.txt --model allenai/scibert_scivocab_uncased --name sci_matsciaug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Mat/train.txt --model allenai/scibert_scivocab_uncased --name sci_mataug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Sci/train.txt --model allenai/scibert_scivocab_uncased --name sci_sciaug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Bert/train.txt --model allenai/scibert_scivocab_uncased --name sci_bertaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug2 --device $device --result_json $result_json
    
    # BERT
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_MatSci/train.txt --model bert-base-uncased --name bert_matsciaug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Mat/train.txt --model bert-base-uncased --name bert_mataug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Sci/train.txt --model bert-base-uncased --name bert_sciaug --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Bert/train.txt --model bert-base-uncased --name bert_bertaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --model bert-base-uncased --name bert_gptaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --model bert-base-uncased --name bert_gptaug2 --device $device --result_json $result_json
    
    # MatBERT
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_MatSci/train.txt --model ../data/model/matbert/ --name matbert_matsciaug  --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Mat/train.txt --model ../data/model/matbert/ --name matbert_mataug  --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Sci/train.txt --model ../data/model/matbert/ --name matbert_sciaug  --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../augmentation/$folder/SOFC_Bert/train.txt --model ../data/model/matbert/ --name matbert_bertaug  --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --model ../data/model/matbert/ --name matbert_gptaug  --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --model ../data/model/matbert/ --name matbert_gptaug2  --device $device --result_json $result_json
    
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --name matsci_base --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --name sci_base --model allenai/scibert_scivocab_uncased --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --name bert_base --model bert-base-uncased --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --name matbert_base --model ../data/model/matbert/ --device $device --result_json $result_json

done
    