from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import numpy as np


def data():
	typeFile = open("../data/comment_type.txt",'r')

	types = typeFile.readlines()
	Y = []
	for line in types:
		tmp = int(line.strip())
		if tmp == 1:
			Y.append(tmp)
		else:
			Y.append(0)
	Y = np.fromiter(Y,int)
	Y = np.reshape(Y,(1526,1))
	typeFile.close()


	inputFile = open("../data/comment_processed_train.txt",'r')
	corpus = inputFile.readlines()
	vectorizer = CountVectorizer(ngram_range=(1, 3),min_df=0.025,stop_words='english')
	X = vectorizer.fit_transform(corpus)
	X = X.toarray()
	inputFile.close()
	print(vectorizer.get_feature_names())
	inputFile = open('../data/comment_processed_all.txt','r')
	corpus = inputFile.readlines()
	X_test = vectorizer.transform(corpus).toarray()
	inputFile.close()

	inputFile = open("../data/code_infor_train.txt",'r')
	lines = inputFile.readlines()
	code_infor = []
	for i in range(0,len(lines)):
		numbers = lines[i].strip().split('\t')
		for number in numbers:
			code_infor.append(number)
	code_infor = np.fromiter(code_infor,float)
	code_infor = np.reshape(code_infor,(len(lines),4))
	X = np.concatenate((X,code_infor),axis=1)
	inputFile.close()
	inputFile = open("../data/code_infor_all.txt",'r')
	lines = inputFile.readlines()
	code_infor = []
	for i in range(0,len(lines)):
		numbers = lines[i].strip().split('\t')
		for number in numbers:
			code_infor.append(number)
	code_infor = np.fromiter(code_infor,float)
	code_infor = np.reshape(code_infor,(len(lines),4))
	X_test = np.concatenate((X_test,code_infor),axis=1)
	inputFile.close()

	inputFile = open("../data/comment_infor_train.txt",'r')
	lines = inputFile.readlines()
	comment_infor = []
	for line in lines:
		features = line.strip().split('\t')
		dic = {}
		dic['property'] = features[0]
		dic['ratio'] = float(features[1])
		dic['lenth'] = int(features[2])
		comment_infor.append(dic)
	vec = DictVectorizer()
	comment_infor = vec.fit_transform(comment_infor).toarray()
	X = np.concatenate((X,comment_infor),axis=1)
	print(vec.get_feature_names())
	inputFile = open("../data/comment_infor_all.txt",'r')
	lines = inputFile.readlines()
	comment_infor = []
	for line in lines:
		features = line.strip().split('\t')
		dic = {}
		dic['property'] = features[0]
		dic['ratio'] = float(features[1])
		dic['lenth'] = int(features[2])
		comment_infor.append(dic)
	comment_infor = vec.transform(comment_infor).toarray()
	X_test = np.concatenate((X_test,comment_infor),axis=1)

	inputFile = open("../data/por_train.txt",'r')
	lines = inputFile.readlines()
	prob_infor = []
	dic = {}
	for line in lines:
		features = line.strip().split('\t')
		dic[int(features[0])] = float(features[1])
	for i in range(0,1526):
		if i in dic:
			prob_infor.append(dic[i])
		else:
			prob_infor.append(0)
	prob_infor = np.fromiter(prob_infor,float)
	prob_infor = np.reshape(prob_infor,(1526,1))
	X = np.concatenate((X,prob_infor),axis=1)
	inputFile = open("../data/pro_all.txt",'r')
	lines = inputFile.readlines()
	prob_infor = []
	dic = {}
	for line in lines:
		features = line.strip().split('\t')
		dic[int(features[0])] = float(features[1])
	for i in range(0,90419):
		if i in dic:
			prob_infor.append(dic[i])
		else:
			prob_infor.append(0)
	prob_infor = np.fromiter(prob_infor,float)
	prob_infor = np.reshape(prob_infor,(90419,1))
	X_test = np.concatenate((X_test,prob_infor),axis=1)
	return Y, X ,X_test




def training(Y,X,X_test):
	clf = RandomForestClassifier(n_estimators=130,min_samples_split=20,max_depth=120,max_features=4,oob_score=True,criterion='gini')
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(80,80,30), random_state=1,activation = 'logistic',max_iter = 200)

	clf = clf.fit(X, Y.ravel())
	print(clf.feature_importances_)


	result = clf.predict(X_test)

	
	outputfile = open('results.txt','w')
	for i in result:
		outputfile.write(str(i)+'\n')
	outputfile.close()
	
	return clf




def ten_folds_test(X,Y):

	clf = RandomForestClassifier(n_estimators=130,min_samples_split=20,max_depth=120,max_features=4,oob_score=True,criterion='gini')
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(80,80,30), random_state=1,activation = 'logistic',max_iter = 200)
	print(cross_val_score(clf, X, Y.ravel(), scoring='precision',cv=10))
	print(cross_val_score(clf, X, Y.ravel(), scoring='recall',cv=10))  
	print(cross_val_score(clf, X, Y.ravel(), scoring='f1',cv=10))    

	


if __name__=="__main__":
	Y,X,X_test = data()
	clf = training(Y,X,X_test)
	ten_folds_test(X,Y)