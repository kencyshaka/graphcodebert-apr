# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import random
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from dataset import *
from model import *
from dataprocessing import *
from pytorch_lightning.loggers import WandbLogger
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters

    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .pkl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .pkl files for this task.")
    parser.add_argument("--val_filename", default=None, type=str,
                        help="The test filename. Should contain the .pkl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .pkl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_data_processing", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_best_bleu", action='store_true',
                        help="Whether to store the best model based on best bleu value")
    parser.add_argument("--do_best_loss", action='store_true',
                        help="Whether to store the best model based on best loss value")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args.seed)

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    # build model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    seq2seq_model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                            beam_size=args.beam_size, max_length=args.max_target_length,
                            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        seq2seq_model.load_state_dict(torch.load(args.load_model_path))  # before it was model

    #    model.to(device)
    #    if args.n_gpu > 1:
    #        # multi-gpu training
    #        model = torch.nn.DataParallel(model)

    if args.do_data_processing:
        train_examples = retrieve_pkl_file(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, args.train_filename,
                                                      stage='train')

    if args.do_train:
        # Prepare training data loader
        train_features = retrieve_pkl_file(args.train_filename)
        val_features = retrieve_pkl_file(args.val_filename)
        dev_features = retrieve_pkl_file(args.dev_filename)
        test_features = retrieve_pkl_file(args.test_filename)
        targets = dev_features[0], test_features[0]

        model = Seq2SeqPredictor(seq2seq_model, tokenizer, targets, args)  # lightning model

        # train_fetaures,val_features = train_test_split(train_features, test_size=0.2, random_state=args.seed, shuffle=True)
        data_module = TextDatasetModule(train_features[:8], val_features[:8], dev_features[:8], test_features[:8], args)

        # for loss function
        best_ppl_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
        if not os.path.exists(best_ppl_checkpoint_dir):
            os.makedirs(best_ppl_checkpoint_dir)

        best_bleu_checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
        if not os.path.exists(best_bleu_checkpoint_dir):
            os.makedirs(best_bleu_checkpoint_dir)

        if args.do_best_bleu:
            dirpath_checkpoint = best_bleu_checkpoint_dir
            metric_value = "avg_val_bleu"
            mode_value = "max"

        if args.do_best_loss:
            dirpath_checkpoint = best_ppl_checkpoint_dir
            metric_value = "avg_val_loss"
            mode_value = "min"

        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath_checkpoint,
            filename="pytorch_model.bin",
            save_top_k=1,
            verbose=True,
            monitor=metric_value,
            mode=mode_value
        )

        lastcheckpoint_dir = os.path.join(args.output_dir, 'checkpoint-last')
        if not os.path.exists(lastcheckpoint_dir):
            os.makedirs(lastcheckpoint_dir)

        early_stop_callback = EarlyStopping(
            monitor=metric_value,
            min_delta=0.00,
            patience=2,
            verbose=True,
            mode=mode_value
        )
        save_dir = os.path.join(args.output_dir, 'lightning_logs')

        # make dir if save_dir not exist
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        wandb_logger = WandbLogger(save_dir=save_dir,
                                   project="graphcodebert_experiment"
                                   )

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            # checkpoint_callback=checkpoint_callback,
            max_epochs=args.num_train_epochs,
            gpus=args.n_gpu,
        )

        # log gradients and model topology
        wandb_logger.watch(model)

        trainer.fit(model, data_module)

    if args.do_test:
        trainer.test()
        # my comment the following can be at the end of testing


#            dev_bleu=round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file),
#                                 os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)),2)
#            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
#            logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
#            logger.info("  "+"*"*20)

if __name__ == "__main__":
    main()
