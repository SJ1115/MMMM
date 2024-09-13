start=`date +%s`

#bash run_SOFC_slot.sh
#bash run_SOFC.sh


folder=v0904
device=1
seed=123

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_MatSci_equal --data matscholar --merge_file data/task/matscholar/train.txt
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_Sci_equal --data matscholar --merge_file data/task/matscholar/train.txt --model allenai/scibert_scivocab_uncased
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_Mat_equal --data matscholar --merge_file data/task/matscholar/train.txt --model ../data/model/matbert/
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_Bert_equal --data matscholar --merge_file data/task/matscholar/train.txt --model bert-base-uncased

for resample in none binomial poisson geometric; do
    #python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_MatSci_$resample --resample_length $resample --data matscholar --merge_file data/task/matscholar/train.txt
    python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_$resample --resample_length $resample --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt  --ner_epochs 40
    python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_$resample --resample_length $resample --data sofc --merge_file data/task/sofc_process/train.txt

done

#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_Bert_equal --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --model bert-base-uncased
#python generate_MELM.py --device $device -fp16 --resample_length binomial --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_Bert_bin --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --model bert-base-uncased
#python generate_MELM.py --device $device -fp16 --resample_length poisson --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_Bert_poi --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --model bert-base-uncased
#python generate_MELM.py --device $device -fp16 --resample_length geometric --seed $seed -save_intermediate --result_dir augmentation/$folder/Slot_Bert_geo --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --model bert-base-uncased
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/SOFC_Bert --data sofc --merge_file data/task/sofc_process/train.txt --model bert-base-uncased
#python generate_MELM.py --device $device -fp16 --seed $seed -save_intermediate --result_dir augmentation/$folder/matsch_Bert --data matscholar --merge_file data/task/matscholar/train.txt --model bert-base-uncased



bash run_SOFC_slot_bin.sh
bash run_SOFC_bin.sh
#bash run_matscholar.sh

### END ###

