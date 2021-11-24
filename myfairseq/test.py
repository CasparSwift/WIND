import os
import argparse

parser = argparse.ArgumentParser("test")
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str, default=None)
parser.add_argument('--init_method', type=str, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--output_path', type=str)


def main():
    args = parser.parse_args()

    input_path = 'cross-domain-en-de-500K'
    output_path = args.output_path

    subset = ['valid1', 'test', 'test1']
    best_bleu = {s: 0 for s in subset}
    best_bleu_ckpt = {s: 0 for s in subset}
    for i in range(12, 36):
        cmd = f'''cd /cache/myfairseq && python scripts/average_checkpoints.py \
            --inputs /cache/myfairseq/checkpoints/{output_path} \
            --num-epoch-checkpoints 5 --checkpoint-upper-bound {i+4} \
            --output {output_path}_averaged_model.pt'''
        print(cmd)
        os.system(cmd)

        ckpt_path = f'{output_path}_averaged_model.pt'

        for s in subset:
            cmd = f'''cd /cache/myfairseq && fairseq-generate \
                /cache/{input_path} --path {ckpt_path} \
                --remove-bpe --beam 4 --batch-size 64 --lenpen 0.6 \
                --max-len-a 1 --max-len-b 50 --gen-subset {s} |tee generate_{s}.out'''
            print(cmd)
            result = os.popen(cmd).read()

            constant = "'s{(\S)-(\S)}{$1 ##AT##-##AT## $2}g'"
            os.system(f"cd /cache/myfairseq && grep ^T generate_{s}.out | cut -f2- | perl -ple {constant} > generate_{s}.ref")
            os.system(f"cd /cache/myfairseq && grep ^H generate_{s}.out | cut -f3- | perl -ple {constant} > generate_{s}.sys")
            res = os.popen(f"cd /cache/myfairseq && fairseq-score -s generate_{s}.sys -r generate_{s}.ref").read()
            bleu = float(res.split('BLEU4')[-1].split(',')[0].split('= ')[-1])
            print(res)
            if bleu > best_bleu[s]:
                best_bleu[s] = bleu
                best_bleu_ckpt[s] = i
        print(best_bleu)
        print(best_bleu_ckpt)


if __name__ == '__main__':
    main()



