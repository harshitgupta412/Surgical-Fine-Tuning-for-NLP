!python3 ft.py \
    --task ft \
    --model sst2 \
    --dataset yelp_polarity,cola,sst2_aug_random_insertion,sst2_aug_random_swap,sst2_aug_synonym_replacement \
    --k 300 \
    --mode all \
    --num_layers 1,2,3 \
    --lora_mode same,0,3,6,9,11 \
    --lora_k 4,16,32,64

!python3 ft.py \
    --task ft \
    --model sst2 \
    --dataset yelp_polarity,cola,sst2_aug_random_insertion,sst2_aug_random_swap,sst2_aug_synonym_replacement \
    --k 300 \
    --mode all,0,3,6,9,11 \
    --num_layers 1,2,3 \
    --lora_mode none \
    --lora_k 4
    
!python3 ft.py \
    --task ft \
    --model sst2 \
    --dataset yelp_polarity,cola,sst2_aug_random_insertion,sst2_aug_random_swap,sst2_aug_synonym_replacement \
    --k 300 \
    --mode 0,3,6,9,11 \
    --num_layers 1,2,3 \
    --lora_mode same \
    --lora_k 32,64,16,4