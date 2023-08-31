import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--agency_list_eval", default='223', type=str)  #  429 397 597
parser.add_argument("--agency_list_train", default='223', type=str)  # 432 223 734 429 830 299 579 527 587 976 832 254 397 42 59 383 967 43 713 465 24 56 597 453 650 110 154 485

# parser.add_argument("--inputs", default='0 1 2 3 4 6 7 8 9 10 11 12 13 14', type=str) # 0 1 2 3 4 6 7 8 9 10 11 12 13 14
# parser.add_argument("--outputs", default='17', type=str)

parser.add_argument("--mt", default='Coronary_Angiography', type=str)  #  Coronary_Angiography  Cardiac_Stent   CABG

parser.add_argument("--version", default='v1', type=str)
# parser.add_argument("--diff_rate", default=0.03, type=float)

parser.add_argument("--train_bool", default=1, type=int)
parser.add_argument("--train_sen_bool", default=1, type=int)

parser.add_argument("--epochs", default=150, type=int)
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument("--learning_rate", default=2e-3, type=float)  #2e-5
parser.add_argument("--dropout", default=0.6, type=float)

parser.add_argument("--embed_dim", default=10, type=int)
parser.add_argument("--conv_layers", default=3, type=int)
parser.add_argument("--conv_nodes", default=32, type=int)
parser.add_argument("--gamma", default=2, type=float)
parser.add_argument("--alpha", default=0.9, type=float)

# parser.add_argument("--start", default=130, type=int)
# parser.add_argument("--end", default=145, type=int)

parser.add_argument("--device", default='/cpu:0', type=str)

args = parser.parse_args()


def load_parameters():
    FIXED_PARAMETERS = {
        "data_path": "./data/split",
        "agency_list_eval": args.agency_list_eval,
        "agency_list_train": args.agency_list_train,
        # "inputs": args.inputs,
        # "outputs": args.outputs,
        "mt": args.mt,
        # "diff_rate": args.diff_rate,
        "train_bool": args.train_bool,
        "train_sen_bool": args.train_sen_bool,
        "print_bool": 1,
        # "NAME": 'weights_'+args.dataset_train+"_"+args.version,
        "version": args.version,
        "epochs": args.epochs,
        "batchsize": args.batchsize,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "embed_dim": args.embed_dim,
        "conv_layers": args.conv_layers,
        "conv_nodes": args.conv_nodes,
        "gamma": args.gamma,
        "alpha": args.alpha,

        # "start": args.start,
        # "end": args.end,
        "device": args.device
    }
    return FIXED_PARAMETERS
