import jsonlines
import os

import argparse

def is_span(answers, passage):
  for ans in answers:
    if ans.strip(". ,;/'?><").lower() in passage['passage'].lower():
      return True, ans

  return False, answers[0]


def split(data_file, qa_dir, nlg_dir, set_type):
  if set_type == "train":
    file_name = 'train.jsonl'
    
  qa_examples = []
  with jsonlines.open(data_file) as reader:
    for example in reader:
      if example['Answer_new'][0] != "No Answer Present.":
        new_example = example.copy()
        golden_passage = new_example['passage'][0]
        golden_answer = new_example['Answer_new'][0]
        for passage in new_example['passage']:
            res, golden_answer = is_span(new_example['Answer_new'], passage)
            if res:
              golden_passage = passage
              break
        golden_passage.pop('url')
        new_example['passage'] = golden_passage
        new_example[ 'Answer_new'] = golden_answer if set_type == "train" else new_example['Answer_new']
        new_example.pop('passage')
        new_example.pop('Answer_new')

        qa_examples.append(new_example)

  nlg_examples = []
  with jsonlines.open(data_file) as reader:
    for example in reader:
        new_example.pop('Answer_new')
        golden_passage = new_example['passage'][0]
        for passage in new_example['passage']:
            golden_passage = passage

        for passage in new_example['passage']:
            res, golden_answer = is_span(passage)
            if res:
              golden_passage = passage
        new_example['passage'] = golden_passage
        new_example['Answer_new'] = golden_answer
        new_example.pop('passage')

        nlg_examples.append(new_example)

  for (new_data_dir, examples) in zip((qa_dir, nlg_dir),
                                      (qa_examples, nlg_examples)):
    new_data_file = os.path.join(new_data_dir, file_name)
    with jsonlines.open(new_data_file, mode='w') as writer:
      writer.write_all(examples)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The input data dir.")
  args = parser.parse_args()  
  dataset_dir = args.data_dir

  qa_dir = os.path.join(dataset_dir, 'qa')
  nlg_dir = os.path.join(dataset_dir, 'nlg')

  if not os.path.exists(qa_dir):
    os.makedirs(qa_dir)
  if not os.path.exists(nlg_dir):
    os.makedirs(nlg_dir)

  train_file = os.path.join(dataset_dir, "train_v2.1.jsonl")
  split(train_file, qa_dir, nlg_dir, 'train')