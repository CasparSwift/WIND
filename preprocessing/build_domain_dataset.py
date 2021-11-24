import os
import argparse

parser = argparse.ArgumentParser("preprocess for iwslt nmt dataset")
parser.add_argument("--src", type=str)
parser.add_argument("--tgt", type=str)
parser.add_argument("--train_year", type=str)
parser.add_argument("--test_year", type=str)
parser.add_argument("--dev", type=str)
parser.add_argument("--test", type=str)
parser.add_argument("--download", action='store_true')
parser.add_argument("--extract", action='store_true')
parser.add_argument("--tokenize", action='store_true')
parser.add_argument("--bpe", action='store_true')
parser.add_argument("--binarize", action='store_true')
parser.add_argument("--output", type=str) # 数据集输出位置
parser.add_argument("--out_of_domain", type=str) # out_of_domain data位置
parser.add_argument("--destdir", type=str, default='data-bin') # binarize之后的输出位置
args = parser.parse_args()

args_dict = {
    'src': args.src,
    'tgt': args.tgt,
    'train_year': args.train_year,
    'test_year': args.test_year,
    'dev': args.dev,
    'output': args.output,
    'test_set': ' '.join('$' + name for name in args.test.split()),
    'test_set_string': '\n'.join(f'{name}={name}' for name in args.test.split()),
    'test_set2': ' '.join(f'test.${name}.$L' for name in args.test.split()),
    'test_set3': ','.join(f'$prep/test.{name}' for name in args.test.split()),
    'destdir': args.destdir,
    'out_of_domain': args.out_of_domain
}

download_string = '''
#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

URL="https://wit3.fbk.eu/archive/20{train_year}-01/texts/{src}/{tgt}/{src}-{tgt}.tgz"
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

GZ={src}-{tgt}.tgz
GZ2={src}-{tgt}-{train_year}.tgz
mv $GZ $GZ2
tar zxvf $GZ2

cd ..
mkdir wmt14_en_de
cd wmt14_en_de
wget http://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget http://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
'''.format(**args_dict)


# extract sentence from xml format
extract_string = '''
#!/usr/bin/env bash
echo "extract train data from xml format..."
src={src}
tgt={tgt}
lang={src}-{tgt}
orig=orig

for l in $src $tgt; do
    cat ./{out_of_domain}/train.$l > $orig/$lang-{train_year}/out_of_domain_train.$l
done

for l in $src $tgt; do
    f=train.tags.$lang.$l
    cat $orig/$lang-{train_year}/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\\/description>//g' > $orig/$lang-{train_year}/in_domain_train.$l
done

cd $orig/$lang-{train_year}

for l in $src $tgt; do
	cat out_of_domain_train.$l in_domain_train.$l > train.$l
	rm out_of_domain_train.$l
    wc -l *train.$l
done

'''.format(**args_dict)


tokenize_string = '''
#!/usr/bin/env bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
# LC=$SCRIPTS/tokenizer/lowercase.perl

GZ={src}-{tgt}.tgz
src={src}
tgt={tgt}
lang={src}-{tgt}
prep={output}
tmp=$prep/tmp
orig=orig

mkdir -p $prep $tmp

echo "pre-processing train data..."
for l in $src $tgt; do
    tok=train.tags.$lang.tok.$l
    cat $orig/$lang-{train_year}/train.$l | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    cat $orig/$lang-{train_year}/in_domain_train.$l | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/in_domain_$tok
    echo ""
done

for l in $src $tgt; do
    # perl $LC < $tmp/train.tags.$lang.tok.$l > $tmp/train.tags.$lang.$l
    mv $tmp/train.tags.$lang.tok.$l $tmp/train.tags.$lang.$l
done

for l in $src $tgt; do
    # perl $LC < $tmp/in_domain_train.tags.$lang.tok.$l > $tmp/in_domain_train.tags.$lang.$l
    mv $tmp/in_domain_train.tags.$lang.tok.$l $tmp/in_domain_train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang-{test_year}/IWSLT{test_year}.TED*.$l.xml`; do
    fname=${{o##*/}}
    f=$tmp/${{fname%.*}}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\\s*//g' | \
        sed -e 's/\\s*<\\/seg>\\s*//g' | \
        sed -e "s/\\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l > $f  #| \
    # perl $LC > $f
    echo ""
    done
done
'''.format(**args_dict)


bpe_string = '''
#!/usr/bin/env bash

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000

src={src}
tgt={tgt}
lang={src}-{tgt}
prep={output}
tmp=$prep/tmp
orig=orig

{test_set_string}

echo "creating train, in-domain train, valid, test..."
for l in $src $tgt; do
    mv $tmp/train.tags.$lang.$l $tmp/train.$l
    mv $tmp/in_domain_train.tags.$lang.$l $tmp/in_domain_train.$l
    mv $tmp/IWSLT{test_year}.TED.{dev}.$src-$tgt.$l $tmp/valid.$l
    for t in {test_set}; do
        mv $tmp/IWSLT{test_year}.TED.$t.$src-$tgt.$l $tmp/test.$t.$l
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
    for f in train.$L in_domain_train.$L valid.$L {test_set2}; do
        echo "apply_bpe.py to ${{f}}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

SCRIPTS=mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.in_domain_train $src $tgt $prep/in_domain_train 1 250

for L in $src $tgt; do
    for f in valid.$L {test_set2}; do
        cp $tmp/bpe.$f $prep/$f
    done
done
'''.format(**args_dict)


binarize_string = '''
#!/usr/bin/env bash
prep={output}
fairseq-preprocess --source-lang {src} --target-lang {tgt} \
    --trainpref $prep/train \
    --validpref $prep/in_domain_train,$prep/valid \
    --testpref {test_set3} \
    --destdir {destdir} \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary
'''.format(**args_dict)


def run_bash(bash_string):
    with open('bash.sh', 'w') as f:
        f.write(bash_string)
    os.system('sh bash.sh')
    os.system('rm bash.sh')


if __name__ == '__main__':
    if args.download:
        run_bash(download_string)
    if args.extract:
        run_bash(extract_string)
    if args.tokenize:
        run_bash(tokenize_string)
    if args.bpe:
        run_bash(bpe_string)
    if args.binarize:
        run_bash(binarize_string)