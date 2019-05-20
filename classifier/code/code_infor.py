
commentfile = open("../data/commentfile.txt_all",'r',encoding="utf8")
codepath = '../data/snippets_all/snippet_'
outputfile = open("../data/code_infor_all.txt",'w')

commentlines = commentfile.readlines()

for line in commentlines:
	number = line.split('\t')[0]
	comment = line.split('\t')[1].strip()
	filename = codepath + number + '.java'
	codefile = open(filename,'r',encoding="utf8")
	codelines = codefile.readlines()
	newcodelines = []
	for l in codelines:
		l = l.strip()
		if l == '':
			continue
		newcodelines.append(l)
	featurelen = len(newcodelines)

	if featurelen <= 1:
		outputfile.write(str(featurelen) + '\t0\t0\t0\n')
		continue

	featurebranch = 0
	if 'if' in newcodelines[0] or 'else' in newcodelines[0] or 'while' in newcodelines[0] or 'case' in newcodelines[0]:
		if '{' in newcodelines[0] or '{' in newcodelines[1]:
			featurebranch = 1

	featuremethod = 0
	if 'public' in newcodelines[0] or 'private' in newcodelines[0]:
		if '(' in newcodelines[0]:
			if '{' in newcodelines[0] or '{' in newcodelines[1]:
				featuremethod = 1

	featurecor = 0
	newcomment = ''
	for i in comment:
		if i == ' ':
			newcomment = newcomment + i
		elif i.isalpha():
			newcomment = newcomment + i.lower()
	words = newcomment.split(' ')
	for l in newcodelines:
		newcode = ''
		for i in l:
			if i == ' ':
				newcode = newcode + i
			elif i.isalpha():
				newcode = newcode + i.lower()
		codewords = newcode.split(' ')
		for word in codewords:
			if word in words:
				featurecor = 1

	outputfile.write(str(featurelen) + '\t' + str(featurebranch) + '\t' + str(featuremethod) + '\t' + str(featurecor) + '\n')

outputfile.close()
commentfile.close()