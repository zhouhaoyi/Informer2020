
import sys
import torch
import numpy as np
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer


if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']



args = dotdict()

args.root_path = 'input/'

args.data_path = 'data.csv'

args.test_size = 0.2

args.path_to_scaler_of_target = None

args.direct_to_replace = False



args.direct_date_column = None

args.direct_target_column = None

args.model = 'informer'                      #  options: [ informer, informerstack-NOTSUREYET!-, informerlight(TBD)-NOTSUREYET!-]

args.data = 'custom'

args.features = 'MS'

args.target = 'Close'

args.freq = 'b'

args.checkpoints = './informer_checkpint' # location of model checkpoints

args.seq_len = 1*5*2
args.label_len = 1*1
args.pred_len = 1*1

args.enc_in  =  6            # Encoder input size
args.dec_in  =  6            # Decoder input size
args.c_out   =  1             # Output size

args.factor  =  1              # Probsparse attn factor

args.d_model = 512             # Dimension of model
args.n_heads = 16               # 16خوب
args.e_layers = 8           # num of encoder layers
args.d_layers = 8           # num of decoder layers
args.d_ff    =  2048         # dimension of fcn in model
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

args.itr = 2

args.train_epochs = 30

args.patience = 10

args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.c = np.float64

args.gpu = 0
args.use_multi_gpu = False
args.devices = '0,1,2,3'


args.kind_of_optim = 'AdamW'                 # optimizer to use, options:  AdamW  |  SparseAdam  |  RMSprop  |  Adagrad
#                                                                          RAdam  |  NAdam  |  LBFGS  |  Adamax
#                                                                       Adadelta | Adam  |  SGD |  ASGD

args.inverse=False

args.scale=True

args.kind_of_scaler = 'MinMax'                       #   Standard   or   MinMax

args.scale_with_a_copy_of_target = True           # Create a copy of Target with another name,
#                                                 Scale it along side others.and then replace its name with actual name of targert

args.dtype_ = None                  #np.float64              #      dtype to load the main data file using it.

args.take_data_instead_of_reading = False            # Defualt to False.. if it is True,


args.shuffle_for_test  =  False
args.shuffle_for_pred  =  False
args.shuffle_for_train  =  True

args.criter =  'mse'           # lose function options:  WMAPE | SMAPE | MAE | RMSE | QuantileLoss | HuberLoss | PinballLoss

#args.save_scaler_object = False
#args.output_distribution_ = 'normal'       # This just work with QuantileTransformer #also could use -> uniform -> less comon


args.detail_freq = args.freq
args.freq = args.freq[-1:]
print('Args in experiment:')
print(args)
Exp = Exp_Informer

temp_ = input(" Please Press Inter To Continue . . .  Or  Type q and then Press Enter to Exit the Progress! ")
if temp_.lower() == 'q':
    exit()
else:
    pass

for ii in range(args.itr):

    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.attn,
        args.factor,
        args.embed,
        args.distil,
        args.mix,
        args.des,
        ii)
    
    # set experiments
    exp = Exp(args)
    
    # train
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    # test
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    
    torch.cuda.empty_cache()

print("Done . ")