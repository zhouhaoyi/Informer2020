### M
itr_num=1
DATASET=ETTm1
DEVICES=0,1,2,3,4,5,6,7

# attn $ATTN
ATTN=$ATTN

python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 48  --label_len 48  --pred_len 24  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 96  --label_len 96  --pred_len 48  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
# attn $ATTN

python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 48  --label_len 48  --pred_len 24  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 96  --label_len 96  --pred_len 48  --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp' --use_multi_gpu --devices $DEVICES  --itr $itr_num



### S

python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES   --itr $itr_num --factor 3

python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 96 --label_len 96 --pred_len 48 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES   --itr $itr_num

python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES   --itr $itr_num

python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 336 --label_len 168 --pred_len 336 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES   --itr $itr_num

python -u main_informer.py --model informer --data ETTh2 --features S --seq_len 336 --label_len 336 --pred_len 720 --e_layers 3 --d_layers 2 --attn $ATTN --des 'Exp'  --use_multi_gpu --devices $DEVICES   --itr $itr_num
