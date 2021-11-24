import random

indexs = set(random.sample(list(range(4468840)), 500000))

with open('wmt14_en_de_500K/train.en', 'a') as fw:
	with open('wmt14_en_de/train.en', 'r') as f:
		for i, line in enumerate(f):
			if i in indexs:
				fw.write(line)

with open('wmt14_en_de_500K/train.de', 'a') as fw:
	with open('wmt14_en_de/train.de', 'r') as f:
		for i, line in enumerate(f):
			if i in indexs:
				fw.write(line)