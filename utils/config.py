import argparse
# Training settings
parser = argparse.ArgumentParser(description='dorefa-net implementation')

#########################
#### data parameters ####
#########################
parser.add_argument("--data_name", type=str, default="wiki", # wiki xmedianet2views nus inria
                    help="data name")
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='partiallabel')
parser.add_argument('--ckpt_dir', type=str, default='partiallabel')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--partial_file', type=str, default="wiki/partial_labels_0.2_sym.json")
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--output_dim', type=int, default=10, help='output shape')
parser.add_argument('--partial_ratio', type=float, default=0.2)
parser.add_argument('--tau', type=float, default=1.)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--optimizer', type=str, default='Adam')#
parser.add_argument('--views', nargs='+', help='<Required> Quantization bits', default=['Img', 'Txt'])
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--loss_weight', default=5.0, type=float,
                    help='contrastive loss weight')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')

parser.add_argument('-lr_decay_epochs', type=int, default=9,
                    help='decay epochs for learning rate')

parser.add_argument('--zeta', type=float, default=1.)

parser.add_argument('--proto_m', type=float, default=1.)
parser.add_argument('--pro_weight_start', default='0.9', type=str,
                        help='prototype updating coefficient')
parser.add_argument('--pro_weight_end', default='0.5', type=str,
                        help='prototype updating coefficient')
parser.add_argument('--class_num', type=float, default=10)
parser.add_argument('--seed', type=int, default=0, help="random seed")

parser.add_argument('--w1', type=float, default=1, help="weight of loss_nbd")
parser.add_argument('--w2', type=float, default=0, help="wight of loss_icc")
parser.add_argument('--w3', type=float, default=0, help="wight of loss_pca")

parser.add_argument('--lamda', default=5, type=float)
parser.add_argument('--method', default='ours', type=str)

args = parser.parse_args()
print(args)
