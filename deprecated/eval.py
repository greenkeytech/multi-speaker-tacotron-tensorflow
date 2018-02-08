import os
import re
import math
import argparse
from glob import glob

from synthesizer import Synthesizer
from train import create_batch_inputs_from_texts
from utils import makedirs, str2bool, backup_file
from hparams import hparams, hparams_debug_string

texts = [
  "The buses aren't the problem; they are the solution.",
  "Luke, I am your father.",
  "Dec eighteen schatz five bid six ask in one yard",
  "Euro five week one fifteen eight and twelve, eight and a quarter.",
  "Variational autoencoders are da bomb.",
]


def get_output_base_path(load_path, eval_dirname="eval"):
  if not os.path.isdir(load_path):
    base_dir = os.path.dirname(load_path)
  else:
    base_dir = load_path

  base_dir = os.path.join(base_dir, eval_dirname)
  if os.path.exists(base_dir):
    backup_file(base_dir)
  makedirs(base_dir)

  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(load_path)
  base_path = os.path.join(base_dir, 'eval-%d' % int(m.group(1)) if m else 'eval')
  return base_path


def run_eval(args):
  print(hparams_debug_string())

  load_paths = glob(args.load_path_pattern)

  for load_path in load_paths:
    if not os.path.exists(os.path.join(load_path, "checkpoint")):
      print(" [!] Skip non model directory: {}".format(load_path))
      continue

    synth = Synthesizer()
    synth.load(load_path)

    for speaker_id in range(synth.num_speakers):
      base_path = get_output_base_path(load_path, "eval-{}".format(speaker_id))

      inputs, input_lengths = create_batch_inputs_from_texts(texts)

      for idx in range(math.ceil(len(inputs) / args.batch_size)):
        start_idx, end_idx = idx * args.batch_size, (idx + 1) * args.batch_size

        cur_texts = texts[start_idx:end_idx]
        cur_inputs = inputs[start_idx:end_idx]

        synth.synthesize(
          texts=cur_texts,
          speaker_ids=[speaker_id] * len(cur_texts),
          tokens=cur_inputs,
          base_path="{}-{}".format(base_path, idx),
          manual_attention_mode=args.manual_attention_mode,
          base_alignment_path=args.base_alignment_path,
        )

    synth.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=16)
  parser.add_argument('--load_path_pattern', required=True)
  parser.add_argument('--base_alignment_path', default=None)
  parser.add_argument(
    '--manual_attention_mode', default=0, type=int, help="0: None, 1: Argmax, 2: Sharpening, 3. Pruning"
  )
  parser.add_argument(
    '--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs'
  )
  args = parser.parse_args()

  #hparams.max_iters = 100
  #hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
