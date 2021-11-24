import os
import argparse

parser = argparse.ArgumentParser("preprocess for wmt nmt dataset")
parser.add_argument("--src", type=str)
parser.add_argument("--tgt", type=str)
parser.add_argument("--year", type=str)
# parser.add_argument("--dev", type=str)
# parser.add_argument("--test", type=str)
parser.add_argument("--download", action='store_true')
parser.add_argument("--tokenize", action='store_true')
parser.add_argument("--bpe", action='store_true')
parser.add_argument("--binarize", action='store_true')
args = parser.parse_args()

args_dict = {
	'src': args.src,
	'tgt': args.tgt,
	'year': args.year,
	'dev': args.dev,
	# 'test_set': ' '.join('$' + name for name in args.test.split()),
	# 'test_set_string': '\n'.join(f'{name}={name}' for name in args.test.split()),
	# 'test_set2': ' '.join(f'test.${name}.$L' for name in args.test.split()),
	# 'test_set3': ','.join(f'$prep/test.{name}' for name in args.test.split())
}

download_string = '''
#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

URL="http://www.statmt.org/europarl/v7/{src}-{tgt}.tgz"
orig=orig

mkdir $orig

echo "Downloading data from ${{URL}}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

GZ={src}-{tgt}.tar
tar xvf $GZ
'''.format(**args_dict)


tokenize_string = '''
#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

GZ={src}-{tgt}.tgz
src={src}
tgt={tgt}
lang={src}-{tgt}
prep=iwslt{year}.tokenized.{src}-{tgt}
tmp=$prep/tmp
orig=orig

mkdir -p $prep $tmp

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
# perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.tok.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT{year}.TED*.$l.xml`; do
    fname=${{o##*/}}
    f=$tmp/${{fname%.*}}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\\s*//g' | \
        sed -e 's/\\s*<\\/seg>\\s*//g' | \
        sed -e "s/\\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done
'''.format(**args_dict)


bpe_string = '''
#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

src={src}
tgt={tgt}
lang={src}-{tgt}
prep=iwslt{year}.tokenized.{src}-{tgt}
tmp=$prep/tmp
orig=orig

{test_set_string}

echo "creating train, valid, test..."
for l in $src $tgt; do
    mv $tmp/train.tags.$src-$tgt.$l $tmp/train.$l
    mv $tmp/IWSLT{year}.TED.{dev}.$src-$tgt.$l $tmp/valid.$l
    for t in {test_set}; do
        mv $tmp/IWSLT{year}.TED.$t.$src-$tgt.$l $tmp/test.$t.$l
    done
done

TRAIN=$tmp/train.{src}-{tgt}
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${{TRAIN}}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L {test_set2}; do
        echo "apply_bpe.py to ${{f}}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
'''.format(**args_dict)


binarize_string = '''
#!/usr/bin/env bash
prep=iwslt{year}.tokenized.{src}-{tgt}
fairseq-preprocess --source-lang {src} --target-lang {tgt} \
    --trainpref $prep/train \
    --validpref $prep/valid \
    --testpref {test_set3}
'''.format(**args_dict)

print(binarize_string)


def run_bash(bash_string):
	with open('bash.sh', 'w') as f:
		f.write(bash_string)
	os.system('sh bash.sh')
	os.system('rm bash.sh')


if __name__ == '__main__':
	if args.download:
		run_bash(download_string)
	if args.tokenize:
		run_bash(tokenize_string)
	if args.bpe:
		run_bash(bpe_string)
	if args.binarize:
		run_bash(binarize_string)
	