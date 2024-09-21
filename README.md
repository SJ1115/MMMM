## MMMM

Code for a submitted paper, "Materials-aware Data Augmentation for Named Entity Recognition." 

### Dataset

To use knowledge graph information, we borrowed **MatKG** and tagged every word, indicating whether the word exists in MatKG as an entity.
We uploaded the processed datasets (for training) in `data/MatKG_tag` with the format below.
```
(word)  (entity)  (tag)
...
Polycrystalline   B-DSC  1
perovskites       B-SPL  1
La2/3BAl/#MnO3    B-MAT  0
with              O      0
...
```

### Usage of MMMM

'generate_MMMM.py' runs MMMM with 4 stages.
1.  Finetune BERT-style model(default:MatSciBERT) to generation model. 
2.  Replace entities by MLM inference.
3.  Finetune BERT-style model(same as 1.) to NER model.
4.  Filter generated sentences.
```
python generate_MMMM.py --device 0 --result_dir result_dir --merge_file ./data/sofc_process/slot_train.txt -save_intermediate
```

### Baselines
Among baselines, we implemented an LLM-based approach, noted as `GPT3.5` in the paper.
We also uploaded the source code and prompts in `GPT35.ipynb`.
