import os
import sys
import torch
import numpy as np
from utils.tools import dotdict

from exp.exp_informer import Exp_Informer

args = dotdict()



args.root_path = '/content/input/'

args.data_path = 'data.csv'

args.model = 'informer'                      #  options: [ informer, informerstack-NOTSUREYET!-, informerlight(TBD)-NOTSUREYET!-]

args.data = 'custom'

args.features = 'MS'

args.target = 'Close'

args.freq = 'b'

args.checkpoints = './informer_checkpint' # location of model checkpoints

args.seq_len = 1*5*2
args.label_len = 1*1
args.pred_len = 1*1

args.enc_in  =  6         		   # Encoder input size
args.dec_in  =  6          		  # Decoder input size
args.c_out   =  1            		 # Output size

args.factor  =  1            		  # Probsparse attn factor
	
args.d_model = 512            		 # Dimension of model
args.n_heads = 16            		   # 16خوب
args.e_layers = 8          		 # num of encoder layers
args.d_layers = 8           		# num of decoder layers
args.d_ff    =  2048         		# dimension of fcn in model
args.s_layers = '3,2,1'			#help='num of stack encoder layers'

args.dropout  =  0.01         # dropout

args.attn = 'full'                  # attention used in encoder, options:[prob, full]
args.embed = 'learned'              # time features encoding, options:[timeF, fixed, learned]
args.activation = 'ReLU'            # activation

args.distil = True                      # whether to use distilling in encoder
args.output_attention = False           # whether to output attention in ecoder
args.mix = True

args.padding = 0

args.batch_size = 16
args.learning_rate = 0.00005

args.loss = 'mse'
args.lradj = 'type1'

args.use_amp = False                    # whether to use automatic mixed precision training

args.num_workers = 2

args.itr = 1

args.train_epochs = 5

args.patience = 10

args.des = 'exp'			#default='test',help='exp description'

args.use_gpu = True if torch.cuda.is_available() else False

args.gpu = 0
args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.do_predict = None			# help='whether to predict unseen future dat



#    #    #    new#    #
args.test_size = 0.2

args.kind_of_optim = 'AdamW'               # optimizer to use,options:AdamW|SparseAdam|RMSprop|AdagradRAdam|NAdam|LBFGS|AdamaxAdadelta|Adam |SGD | ASGD

args.inverse=False

args.scale=True

args.kind_of_scaler = 'MinMax'                       #   Standard   or   MinMax

args.take_data_instead_of_reading = False            # Defualt to False.. if it is True, you should provide direct_data!
args.direct_data = None                        

args.shuffle_for_test  =  False
args.shuffle_for_pred  =  False
args.shuffle_for_train  =  False

args.criter =  'mse'       # lose function options: WMAPE|SMAPE|MAE|RMSE|QuantileLoss|HuberLoss|PinballLoss
#    #    #    new#    #



args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

#if args.use_gpu and args.use_multi_gpu:
#    args.devices = args.devices.replace(' ','')
#    device_ids = args.devices.split(',')
#    args.device_ids = [int(id_) for id_ in device_ids]
#    args.gpu = args.device_ids[0]


print('Args in experiment:')
print(args)

Exp = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
