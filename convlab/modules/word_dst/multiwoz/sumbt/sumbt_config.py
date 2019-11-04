
from dotmap import DotMap


args = DotMap()

args.max_label_length = 32
args.max_turn_length = 22
args.hidden_dim = 100
args.num_rnn_layers = 1
args.zero_init_rnn = False
args.attn_head = 4
args.do_eval = True
args.do_train = False
args.do_lower_case = False
args.distance_metric = 'cosine'
args.train_batch_size = 4
args.dev_batch_size = 1
args.eval_batch_size  = 16
args.learning_rate = 5e-5
args.num_train_epochs = 3
args.patience = 10
args.warmup_proportion = 0.1
args.local_rank = -1
args.seed = 42
args.gradient_accumulation_steps = 1
args.fp16 = False
args.loss_scale = 0
args.do_not_use_tensorboard = False

args.fix_utterance_encoder = False
args.do_eval = True
args.no_cuda = True
args.num_train_epochs = 300
args.data_dir = 'data/sumbt/data'
args.bert_dir = '/mnt/c/Users/bapeng/.pytorch_pretrained_bert'
args.bert_model = 'bert-base-uncased'
args.do_lower_case = True
args.task_name = 'bert-gru-sumbt'
args.nbt = 'rnn'
args.output_dir = 'data/sumbt/model'
args.target_slot = 'all'
args.learning_rate = 5e-5
args.train_batch_size = 4
args.eval_batch_size = 16
args.distance_metric = 'euclidean'
args.patience = 15
args.tf_dir = 'tensorboard'
args.hidden_dim = 300
args.max_label_length = 32
args.max_seq_length = 64
args.max_turn_length = 22