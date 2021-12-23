### M
itr_num=1
DATASET=ETTm1
DEVICES=1,2,3,4,5,6,7

# attn prob
ATTN=prob
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 672 --label_len 96  --pred_len 24  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 96  --label_len 48  --pred_len 48  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 384 --label_len 384 --pred_len 96  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num

# attn log
ATTN=prob
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 672 --label_len 96  --pred_len 24  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 96  --label_len 48  --pred_len 48  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 384 --label_len 384 --pred_len 96  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num
python -u main_informer.py --model informer --data $DATASET --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES --itr $itr_num



### S

python -u main_informer.py --model informer --data $DATASET --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES  --itr $itr_num

python -u main_informer.py --model informer --data $DATASET --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES  --itr $itr_num

python -u main_informer.py --model informer --data $DATASET --features S --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES  --itr $itr_num

python -u main_informer.py --model informer --data $DATASET --features S --seq_len 384 --label_len 384 --pred_len 288 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES  --itr $itr_num

python -u main_informer.py --model informer --data $DATASET --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES  --itr $itr_num
