### M
python -u main_informer.py --model informer --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 24 --seq_len 168 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features M --attn prob --d_layers 1 --e_layers 2 --itr 3 --label_len 96 --pred_len 48 --seq_len 96 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 168 --seq_len 336 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 336 --seq_len 720 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 336 --pred_len 720 --seq_len 720 --des 'Exp'

### S
python -u main_informer.py --model informer --data WTH --features S --attn prob --d_layers 1 --e_layers 2 --itr 3 --label_len 168 --pred_len 24 --seq_len 720 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features S --attn prob --d_layers 1 --e_layers 2 --itr 3 --label_len 168 --pred_len 48 --seq_len 720 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features S --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 168 --seq_len 168 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features S --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 336 --pred_len 336 --seq_len 336 --des 'Exp'

python -u main_informer.py --model informer --data WTH --features S --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 336 --pred_len 720 --seq_len 720 --des 'Exp'