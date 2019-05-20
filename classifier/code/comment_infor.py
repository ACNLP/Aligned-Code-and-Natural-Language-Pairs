import nltk

openfile = open("../data/commentfile_all.txt",'r',encoding="utf8")
outputfile = open("../data/comment_infor_all.txt",'w')


lines = openfile.readlines()

for line in lines:
	sent = line.split('\t')[1].lower().strip()
	newsent = ""
	for char in sent:
		if str.isalnum(char) or char== ' ':
			newsent = newsent + char
	if len(sent) == 0:
		ratio = 1
	else:
		ratio = 1 - (float)(len(newsent)/len(sent))
	words = nltk.word_tokenize(newsent)
	word_tag = nltk.pos_tag(words)

	try:
		outputfile.write(word_tag[0][1] + '\t' + str(ratio) + '\t' + str(len(words)) +'\n')
	except IndexError:
		outputfile.write('UNK' + '\t' + str(ratio) + '\t' + str(len(words)) +'\n')
	outputfile.flush()

openfile.close()
outputfile.close()