python3 ft.py \
    --task ft \
    --model bert-tiny \
    --dataset yelp_polarity,cola,sst2,sst2_aug_random_insertion,sst2_aug_random_swap,sst2_aug_synonym_replacement,sst2_aug \
    --k 300,500 \
    --mode all,0,3,6,9,11 \
    --num_layers 1,2,3 \
    --lora_mode all,none,same \
    --lora_k 4,16
