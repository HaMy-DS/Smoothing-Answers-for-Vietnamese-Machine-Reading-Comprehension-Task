import argparse

import os
import logging
import random

import numpy as np

import torch

from train import train

from transformers import BertConfig, BertTokenizer

from model import MUSTTransformerModel
from eval import evaluate
from predict import predict

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)


def main():
  parser = argparse.ArgumentParser()

  # Basic parameters
  
  parser.add_argument("--do_train",
                      action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval",
                      action='store_true',
                      help="Whether to run eval on the eval set.")
  parser.add_argument("--do_pred",
                      action='store_true',
                      help="Whether to run prediction on the  set")
  parser.add_argument("--evaluate_during_training",
                      action='store_true',
                      help="Run evaluation during training\
                            at each evaluation step.")
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The input data dir.")
  parser.add_argument("--train_file",
                      default="train.jsonl",
                      type=str,
                      help="The train file")
  parser.add_argument("--eval_file",
                      default="dev.jsonl",
                      type=str,
                      help="The file for evaluation.")
  parser.add_argument("--pred_file",
                      default="test.jsonl",
                      type=str,
                      help="The file for prediction.")
  parser.add_argument("--span_annotation_file",
                      type=str,
                      help="Annotation file for answer spans.")
  parser.add_argument("--reference_file",
                      default="dev_ref.json",
                      type=str,
                      help="Reference file to get evaluation results.")
  parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The output directory where the model \
                            predictions and checkpoints will be written.")
  parser.add_argument('--overwrite_output_dir',
                      action='store_true',
                      help="Overwrite the content of the output directory")

  # Model parameters
  parser.add_argument("--model_name_or_path",
                      default=None,
                      type=str,
                      required=True,
                      help="Path to pre-trained model or shortcut name")
  parser.add_argument("--config_name",
                      default="",
                      type=str,
                      help="Pretrained config name or path\
                            if not the same as model_name")
  parser.add_argument("--tokenizer_name",
                      default="",
                      type=str,
                      help="Pretrained tokenizer name or path\
                            if not the same as model_name")
  parser.add_argument("--max_num_spans",
                      default=5,
                      type=int,
                      required=True,
                      help="Number of ans spans for each example")
  parser.add_argument("--ed_threshold",
                      default=5,
                      type=int)
  parser.add_argument("--max_seq_len",
                      default=256,
                      type=int,
                      required=True,
                      help="The maximum total input sequence length after\
                            tokenization. Sequences longer than this will \
                            be truncated, sequences shorter will be padded.")

  # Training/Prediction parameters
  parser.add_argument("--per_gpu_train_batch_size",
                      default=8,
                      type=int,
                      help="Batch size per GPU/CPU for training.")
  parser.add_argument("--per_gpu_pred_batch_size",
                      default=8,
                      type=int,
                      help="Batch size per GPU/CPU for prediction.")
  parser.add_argument("--learning_rate",
                      default=5e-5,
                      type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay",
                      default=0.1,
                      type=float,
                      help="Weight decay if we apply some.")
  parser.add_argument("--adam_epsilon",
                      default=1e-6,
                      type=float,
                      help="Epsilon for Adam optimizer.")
  parser.add_argument("--adam_beta1",
                      default=0.9,
                      type=float,
                      help="Beta 1 for Adam optimizer.")
  parser.add_argument("--adam_beta2",
                      default=0.999,
                      type=float,
                      help="Beta 1 for Adam optimizer.")
  parser.add_argument("--max_grad_norm",
                      default=1.0,
                      type=float,
                      help="Max gradient norm.")
  parser.add_argument("--num_train_epochs",
                      default=3.0,
                      type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--max_steps",
                      default=-1,
                      type=int,
                      help="If > 0: set total number of training steps\
                            to perform. Override num_train_epochs.")
  parser.add_argument("--warmup_rate",
                      default=0.0,
                      type=float,
                      help="Linear warmup rate.")
  parser.add_argument('--logging_steps',
                      type=int,
                      default=100,
                      help="Log every X updates steps.")
  parser.add_argument('--eval_steps',
                      type=int,
                      default=50,
                      help="Evaluate every X updates steps.")
  parser.add_argument("--no_cuda",
                      type=bool,
                      default=False,
                      help="Avoid using CUDA when available")
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help="random seed for initialization")

  args = parser.parse_args()

  if os.path.exists(args.output_dir) and os.listdir(args.output_dir)\
          and args.do_train and not args.overwrite_output_dir:
    raise ValueError("Output directory ({}) already exists and is not empty."
                     " Use --overwrite_output_dir to overcome.".format(
                         args.output_dir))

  # Setup CUDA, GPU
  device = torch.device(
      "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
  args.n_gpu = torch.cuda.device_count()
  args.device = device

  # Setup logging
  logging.basicConfig(
      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
      datefmt='%m/%d/%Y %H:%M:%S',
      level=logging.INFO)
  logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

  # Set seed
  # Added here for reproducibility (even between python 2 and 3)
  set_seed(args.seed, args.n_gpu)

  config_class, model_class, tokenizer_class = BertConfig, MUSTTransformerModel, BertTokenizer
  config = config_class.from_pretrained(
      args.config_name if args.config_name else args.model_name_or_path)
  tokenizer = tokenizer_class.from_pretrained(
      args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
  model = model_class.from_pretrained(
      args.model_name_or_path,
      from_tf=bool(".ckpt" in args.model_name_or_path),
      config=config,
      max_seq_len=args.max_seq_len,
      max_num_spans=args.max_num_spans + 1)

  if args.n_gpu > 1 and not args.no_cuda:
    model = torch.nn.DataParallel(model)
  model.to(args.device)

  logger.info("Training/evaluation/prediction hyperparameters %s", args)

  # Training
  if args.do_train:
    global_step, tr_loss = train(args, model, tokenizer, logger)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Evaluation
  if args.do_eval:
    evaluate(args, model, tokenizer, logger)

  # Prediction
  if args.do_pred:
    predict(args, model, tokenizer, logger, args.pred_file, "test",
            "test_prediction.json")


if __name__ == "__main__":
  main()