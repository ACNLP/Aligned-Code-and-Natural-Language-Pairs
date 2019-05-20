from pattern.en import lemma

openfile = open("commentfile.txt",'r')
outputfile = open("comment_processed.txt",'w')

word_bag = {}

lines = openfile.readlines()

for line in lines:
	sent = line.split('\t')[1]
	newsent = ""
	for char in sent:
		if str.isalnum(char) or char== ' ':
			newsent = newsent + char
	words = newsent.split(' ')
	results = ''
	for word in words:
		try:
			word = lemma(word.lower())
			results = results + word + ' '
			if word in word_bag:
				word_bag[word] = word_bag[word] + 1
			else:
				word_bag[word] = 1
		except StopIteration:
			pass
	outputfile.write(results+'\n')
	outputfile.flush()

openfile.close()
outputfile.close()

stopwords = []

openfile = open("stopwords.txt",'r')
lines = openfile.readlines()
for line in lines:
	stopwords.append(line.strip())

results = sorted(word_bag.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
for result in results:
	if result[0] in stopwords or result[1] < 30:
		pass
	else:
		print result

openfile.close()