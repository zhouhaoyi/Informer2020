python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --CSP --itr 10
exit
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --dilated --itr 10
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --passthrough --itr 10
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --CSP --dilated --passthrough --itr 10
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 10

python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn log --des 'Exp' --CSP --itr 10
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn log --des 'Exp' --dilated --itr 10
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn log --des 'Exp' --passthrough --itr 10
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn log --des 'Exp' --CSP --dilated --passthrough --itr 10
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn log --des 'Exp' --itr 10

