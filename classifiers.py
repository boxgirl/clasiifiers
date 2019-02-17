import csv
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,StratifiedShuffleSplit
from sklearn import tree,svm,neighbors,naive_bayes,metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from pprint import pprint
from pathlib import Path
from typing import Any
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA,SparsePCA
import graphviz 
from typing import List,Dict


models=[naive_bayes.GaussianNB(),tree.DecisionTreeClassifier()]
mode=['Validation','Test']
f_path=Path('column_3C_weka.csv')
VALIDATION_FOLDERS=5
TEST_FOLDERS=2
h = .02
class BaseClassifier:

	def __init__(self,file_path:Path):
		self.data= pd.read_csv(file_path)
		le = LabelEncoder().fit(self.data[self.data.columns[-1]].unique())
		sc = StandardScaler()
		self.X=self.data.drop(columns=self.data.columns[-1])
		self.X = SparsePCA(n_components=3).fit_transform(self.X)
		sc.fit(self.X)
		self.X=sc.transform(self.X)
		self.Y= le.transform(self.data[self.data.columns[-1]])
		self.data[self.data.columns[-1]]=le.transform(self.data[self.data.columns[-1]])
		self.classes = pd.Series(self.Y).unique()


	def visualization(self):
		sns.pairplot(self.data)
		df_corr=self.data.corr()
		mask = np.zeros_like(df_corr, dtype=np.bool)
		mask[np.triu_indices_from(mask)] = True
		plt.subplots(figsize = (10,10))
		sns.heatmap(df_corr, 
		            annot=True,
		            mask = masks,
		            cmap = 'RdBu_r',
		            linewidths=0.1, 
		            linecolor='white',
		            vmax = .9,
		            square=True)
		plt.show()

	def classes_viz(self):
		x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
		y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5

		
		plt.figure(2, figsize=(8, 6))
		plt.clf()

		plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap=plt.cm.Set1,
		            edgecolor='k')
		plt.xlabel('sparsePCA_1')
		plt.ylabel('sparsePCA_2')

		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.xticks(())
		plt.yticks(())

		fig = plt.figure(1, figsize=(8, 6))
		ax = Axes3D(fig, elev=-150, azim=110)
		ax.scatter(self.X[:, 0], self.X[:, 1],self.X[:, 2], c=self.Y,
		           cmap=plt.cm.Set1, edgecolor='k', s=40)
		ax.set_title("First three PCA directions")
		ax.set_xlabel("1st eigenvector")
		ax.w_xaxis.set_ticklabels([])
		ax.set_ylabel("2nd eigenvector")
		ax.w_yaxis.set_ticklabels([])
		ax.set_zlabel("3rd eigenvector")
		ax.w_zaxis.set_ticklabels([])
		plt.show()
		
	def check_init(self):
		if not self._initialized:
			raise Exception("Not trained!")

	def train_classifier(self, X, Y):
		raise NotImplementedError()

	def predict(self, data):
		raise NotImplementedError()


class GaussianClassifier(BaseClassifier):
	'''
	GAUSSIAN
	Continuous data

	p(C_k|x)=(p(C_k)p(x|C_k))/p(x)=(prior*likelihood)/evidence=predict/evidence
	prior=class's appearance probability
	likelihood= probability density of feature in C_k
	predict=sum(prior*mult(likelihhod))=sum(p(C_i)mult(p(x_k|C_i)))
	evidence=p(x) (probability of appearance of this feature in classes, it's a sum of their, because it's 'or')
	'''
	def __init__(self,file_path:Path):
		super().__init__(file_path)
		self._initialized = False

	@staticmethod
	def likelihood(X:pd.core.frame.DataFrame, mean:pd.core.series.Series, variance:pd.core.series.Series):
		return (1 / np.sqrt(2 * np.pi * variance)) * np.exp((-(X - mean) ** 2) / (2 * variance))

	def train_classifier(self, X:np.ndarray, Y:np.ndarray):
		self.var = {}
		self.mean = {}
		self.priors = {}
		count = X.shape[0]
		for l in self.classes:
			self.var[l] = X[Y == l].var()
			self.mean[l] = X[Y == l].mean()
			self.priors[l] = Y[Y == l].count() / count
		self._initialized = True

	def predict(self, data:pd.core.frame.DataFrame):
		self.check_init()
		predict = []
		for l in self.classes:
			predict.append(self.priors[l]* GaussianClassifier.likelihood(data, self.mean[l], self.var[l]).product(axis=1))
		df=pd.DataFrame(data=predict)
		return  self.classes[df.values.argmax(axis=0)]

def mygi(Ys:List):
	count = sum(list(map(len,Ys)))
	gini = 0
	for Y in Ys:
		size = Y.shape[0]
		if size == 0:
			continue
		score = np.bincount(Y) / float(size)
		score = np.sum(score * score)
		gini += (1.0 - score) * (size / count)
	return gini

class DecisionTreeClassifier(BaseClassifier):
	
	def __init__(self,file_path:Path):
		super().__init__(file_path)
		self._initialized = False
	
	def predict_row(self,tree:Dict,row:pd.core.series.Series):
		if row[tree['index']] < tree['value']:
				if isinstance(tree['left'], dict):
					return self.predict_row(tree['left'], row)
				else:
					return tree['left']
		else:
				if isinstance(tree['right'], dict):
					return self.predict_row(tree['right'], row)
				else:
					return tree['right']

	def predict(self,test:pd.core.frame.DataFrame):
		self.check_init()
		res=[]
		for idx, row in test.iterrows():
			res.append(self.predict_row(self.tree,row))
		return res

	def get_split(self,X_train:pd.core.frame.DataFrame,y_train:pd.core.series.Series):
		b_index, b_value, b_score, b_groups = 999, 999, 999, []
		for index, column in enumerate(X_train.columns):
			cdata = X_train[column].sort_values()
			Y = y_train[cdata.index]
			cdata = cdata.reset_index(drop=True)
			Y = Y.reset_index(drop=True)
			cdata = cdata.values
			for idx in range(cdata.shape[0]):
				gini = mygi([Y[idx:], Y[:idx]])
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, cdata[idx], gini, [X_train[:idx], Y[:idx], X_train[idx:], Y[idx:]]
		# absolute best 
		return {'index':b_index, 'value':b_value, 'groups':b_groups}
	
	def to_terminal(self,group:pd.core.series.Series):
		return np.argmax(np.bincount(group))
	 
	def split(self,node:Dict, depth:int):
		left,y_left, right,y_right = node['groups']
		del(node['groups'])
		# check for a no split
		if left.shape[0] == 0 or right.shape[0] == 0:
			y_gen=y_left.append(y_right)
			node['left'] = node['right'] = self.to_terminal(y_gen)
			return
		if depth >= self.max_depth:
			node['left'], node['right'] = self.to_terminal(y_left), self.to_terminal(y_right)
			return
		if len(left) <= self.min_size:
			node['left'] = self,to_terminal(y_left)
		else:
			node['left'] = self.get_split(left,y_left)
			self.split(node['left'], depth+1)
		if len(right) <= self.min_size:
			node['right'] = self.to_terminal(y_right)
		else:
			node['right'] = self.get_split(right,y_right)
			self.split(node['right'], depth+1)

	
	def train_classifier(self,X:pd.core.frame.DataFrame,Y:pd.core.series.Series):
		train=np.c_[X,Y]
		self.max_depth=2
		self.min_size=1
		self.tree = self.build_tree(X,Y)
		self._initialized = True

	def build_tree(self,X_train:pd.core.frame.DataFrame,y_train:pd.core.series.Series):
		root = self.get_split(X_train,y_train)
		self.split(root, 1)
		return root



def get_standart_res(model:Any,X_train:pd.core.frame.DataFrame,y_train:pd.core.series.Series,X_test:pd.core.frame.DataFrame,y_test:pd.core.series.Series):
	y_pred = model.fit(X_train, y_train).predict(X_test)
	return metrics.accuracy_score(y_test,y_pred)

def stratified_split(n_folders:int,mode:str,model_standart:Any,my_model:Any, X:np.ndarray, Y:np.ndarray):
	skf=StratifiedShuffleSplit(n_splits=n_folders, test_size=0.33)
	#skf=StratifiedKFold(n_splits=N_FOLDERS)
	my_res=[]
	std_res=[]
	y_pred=0
	X_test=0
	y_test=0
	for train_index, test_index in skf.split(X, Y):
		y_train, y_test = pd.Series(Y[train_index]),pd.Series(Y[test_index])
		X_train, X_test = pd.DataFrame(X[train_index]),pd.DataFrame(X[test_index])
		my_model.train_classifier(X_train, y_train)
		y_pred = my_model.predict(X_test)
		my_res.append(metrics.accuracy_score(y_test, y_pred))
		std_res.append(get_standart_res(model_standart,X_train,y_train,X_test,y_test))
		
	print(mode)
	print('My Res {}: {}'.format(model_standart,np.mean(my_res)))
	print('Std Res {}: {}'.format(model_standart,np.mean(std_res)))

	return np.asarray(X_test),np.asarray(y_pred),np.asarray(y_test)

my_models=[GaussianClassifier(f_path),DecisionTreeClassifier(f_path)]
for my_m,m in zip(my_models,models):
	bc=my_m
	stratified_split(VALIDATION_FOLDERS,mode[0],m,bc,bc.X,bc.Y)
	bc.X,bc.Y,real_Y=stratified_split(TEST_FOLDERS,mode[1],m,bc,bc.X,bc.Y)
bc.classes_viz()
