device=1
melm_seed=123
folder=v0701
result_json=score/slot_tune_$folder.json
#mother=../data/model/matbert/


for Mp in .3 .5 .7; do
    for KGp in .3 .5; do
        for Gp in .5; do
            for Gdist in binomial poisson geometric; do
                for Ml in 1e-5; do
                    for Me in 20 40; do
                        for Ne in 20; do
                            for round in 3 5; do
                                python generate_MELM.py --seed $melm_seed -fp16 --mlm_lr $Ml --mlm_epochs $Me --mlm_p $Mp --KG_p $KGp --rounds $round --gen_p $Gp --ner_epochs $Ne --resample_length $Gdist --device $device -save_intermediate --result_dir augmentation/$folder/Slot_tune/M_l{$Ml}_e{$Me}_p{$Mp}_kg{$KGp}_R{$round}_p{$Gp}_Ne{$Ne}_dist{$Gdist} --data sofc_slot --merge_file data/task/sofc_process/slot_train.txt --ner_src data/model/Slot_MatSci

                                for seed in {1..3}; do
                                    python run_SOFC_NER.py -slot --epochs 40 -fp16 --seed $seed --augment ../augmentation/$folder/Slot_tune/M_l{$Ml}_e{$Me}_p{$Mp}_kg{$KGp}_R{$round}_p{$Gp}_Ne{$Ne}_dist{$Gdist}/train.txt --name M_e{$Me}_p{$Mp}_kg{$KGp}_R{$round}_dist{$Gdist} --device $device --result_json $result_json
                                    python run_SOFC_NER.py -slot --epochs 40 -fp16 --seed $seed --augment ../augmentation/$folder/Slot_tune/M_l{$Ml}_e{$Me}_p{$Mp}_kg{$KGp}_R{$round}_p{$Gp}_Ne{$Ne}_dist{$Gdist}/train.txt --name M_e{$Me}_p{$Mp}_kg{$KGp}_R{$round}_dist{$Gdist} --device $device --result_json $result_json
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done