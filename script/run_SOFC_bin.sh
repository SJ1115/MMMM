
device=1
folder=v0904
result_json=./score/result_sofc_$folder.json


for seed in {1..3}; do
    # MatSciBERT
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_none/train.txt --name matsci_eq --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_binomial/train.txt --name matsci_bin --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_poisson/train.txt --name matsci_poi --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_geometric/train.txt --name matsci_geo --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --name matsci_gptaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --name matsci_gptaug2 --device $device --result_json $result_json
    
    # SciBERT
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_none/train.txt --model allenai/scibert_scivocab_uncased --name sci_eq --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_binomial/train.txt --model allenai/scibert_scivocab_uncased --name sci_bin --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_poisson/train.txt --model allenai/scibert_scivocab_uncased --name sci_poi --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_geometric/train.txt --model allenai/scibert_scivocab_uncased --name sci_geo --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --model allenai/scibert_scivocab_uncased --name sci_gptaug2 --device $device --result_json $result_json
    
    # BERT
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_none/train.txt --model bert-base-uncased --name bert_eq --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_binomial/train.txt --model bert-base-uncased --name bert_bin --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_poisson/train.txt --model bert-base-uncased --name bert_poi --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_geometric/train.txt --model bert-base-uncased --name bert_geo --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --model bert-base-uncased --name bert_gptaug --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --model bert-base-uncased --name bert_gptaug2 --device $device --result_json $result_json
    
    # MatBERT
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_none/train.txt --model ../data/model/matbert/ --name matbert_eq  --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_binomial/train.txt --model ../data/model/matbert/ --name matbert_bin  --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_poisson/train.txt --model ../data/model/matbert/ --name matbert_poi  --device $device --result_json $result_json
    python run_SOFC_NER.py -detail -fp16 --epochs 20 --seed $seed --augment ../augmentation/$folder/SOFC_geometric/train.txt --model ../data/model/matbert/ --name matbert_geo  --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../SOFC_GPT35/train.txt --model ../data/model/matbert/ --name matbert_gptaug  --device $device --result_json $result_json
    #python run_SOFC_NER.py -detail -fp16 -crf --seed $seed --augment ../Sofc_llmda/train.txt --model ../data/model/matbert/ --name matbert_gptaug2  --device $device --result_json $result_json
    
    
done
    