
device=1
seed=123
folder=v0911
# 0 0 1
for k in 100 200 400; do
    ## Aug
    #python generate_MELM.py --device $device --seed $seed -fp16 -save_intermediate --data sofc --mlm_epochs 20 --ner_epochs 20 --merge_file data/restricted/sofc_$k.txt --data_src ../data/restricted/sofc_$k.txt --result_dir augmentation/$folder/restrict/SOFC/$k
    #python generate_MELM_old.py --device $device --seed $seed -fp16 -save_intermediate --data sofc --mlm_epochs 20 --ner_epochs 20 --merge_file data/restricted/sofc_$k.txt --data_src ../data/restricted/sofc_$k.txt --result_dir augmentation/$folder/restrict/SOFC/MELM_$k 
    
    ## Aug MatScholar
    #python generate_MELM.py --device $device --seed $seed -fp16 -save_intermediate --data matscholar --mlm_epochs 20 --ner_epochs 20 --merge_file data/restricted/matsch_$k.txt --data_src ../data/restricted/matsch_$k.txt --result_dir augmentation/$folder/restrict/matsch/$k
    #python generate_MELM_old.py --device $device --seed $seed -fp16 -save_intermediate --data matscholar --mlm_epochs 20 --ner_epochs 20 --merge_file data/restricted/matsch_$k.txt --data_src ../data/restricted/matsch_$k.txt --result_dir augmentation/$folder/restrict/matsch/MELM_$k 
    
    ## Aug Slot
    #python generate_MELM.py --device $device --seed $seed -fp16 -save_intermediate --merge_file data/restricted/slot_$k.txt --mlm_epochs 40 --ner_epochs 40 --data_src ../data/restricted/slot_$k.txt --result_dir augmentation/$folder/restrict/Slot/$k
    #python generate_MELM_old.py --device $device --seed $seed -fp16 -save_intermediate --merge_file data/restricted/slot_$k.txt --mlm_epochs 40 --ner_epochs 40 --data_src ../data/restricted/slot_$k.txt --result_dir augmentation/$folder/restrict/Slot/MELM_$k
    0
done




# 1 0 0 
bash run_matscholar_k.sh
bash run_SOFC_slot_k.sh
bash run_SOFC_k.sh

### END ###