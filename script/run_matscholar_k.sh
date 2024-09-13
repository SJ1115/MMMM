
device=1
folder=v0911
result_json=./score/result_matsch_restricted_$folder.json
for k in 100 200 400; do

    for seed in {1..3}; do
        # MatSciBERT
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --augment ../data/restricted/nomark_matsch_$k.txt --name matsci_base_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --augment ../augmentation/$folder/restrict/matsch/$k/train.txt --name matsci_ours_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --augment ../augmentation/$folder/restrict/matsch/MELM_$k/train.txt --name matsci_MELMMS_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --augment ../matsch_GPT35_base/$k.txt --name matsci_GPT_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --augment ../saved_entda/matsch/train_new_$k.txt --name matsci_ENTDA_$k --device $device --result_json $result_json
        
        # SciBERT
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model allenai/scibert_scivocab_uncased --augment ../data/restricted/nomark_matsch_$k.txt --name sci_base_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/restrict/matsch/$k/train.txt --name sci_ours_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model allenai/scibert_scivocab_uncased --augment ../augmentation/$folder/restrict/matsch/MELM_$k/train.txt --name sci_MELMMS_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model allenai/scibert_scivocab_uncased --augment ../matsch_GPT35_base/$k.txt --name sci_GPT_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model allenai/scibert_scivocab_uncased --augment ../saved_entda/matsch/train_new_$k.txt --name sci_ENTDA_$k --device $device --result_json $result_json
        
        # BERT
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model bert-base-uncased --augment ../data/restricted/nomark_matsch_$k.txt --name bert_base_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model bert-base-uncased --augment ../augmentation/$folder/restrict/matsch/$k/train.txt --name bert_ours_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model bert-base-uncased --augment ../augmentation/$folder/restrict/matsch/MELM_$k/train.txt --name bert_MELMMS_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model bert-base-uncased --augment ../matsch_GPT35_base/$k.txt --name bert_GPT_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model bert-base-uncased --augment ../saved_entda/matsch/train_new_$k.txt --name bert_ENTDA_$k --device $device --result_json $result_json
        
        # MatBERT
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model ../data/model/matbert/ --augment ../data/restricted/nomark_matsch_$k.txt --name mat_base_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model ../data/model/matbert/ --augment ../augmentation/$folder/restrict/matsch/$k/train.txt --name mat_ours_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model ../data/model/matbert/ --augment ../augmentation/$folder/restrict/matsch/MELM_$k/train.txt --name mat_MELMMS_$k --device $device --result_json $result_json
        python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model ../data/model/matbert/ --augment ../matsch_GPT35_base/$k.txt --name mat_GPT_$k --device $device --result_json $result_json
        #python run_matscholar_NER.py -fp16 -crf -detail --seed $seed --model ../data/model/matbert/ --augment ../saved_entda/matsch/train_new_$k.txt --name mat_ENTDA_$k --device $device --result_json $result_json
        
        
    done
done    
    
