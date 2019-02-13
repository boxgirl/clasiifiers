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
from typing import List


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

		# Plot the training points
		plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap=plt.cm.Set1,
		            edgecolor='k')
		plt.xlabel('sparsePCA_1')
		plt.ylabel('sparsePCA_2')

		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.xticks(())
		plt.yticks(())

		plt.figure(2, figsize=(8, 6))
		plt.clf()

		# Plot the training points
		plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap=plt.cm.Set1,
		            edgecolor='k')
		plt.xlabel('sparsePCA_1')
		plt.ylabel('sparsePCA_2')

		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.xticks(())
		plt.yticks(())

		# To getter a better understanding of interaction of the dimensions
		# plot the first three PCA dimensions
		fig = plt.figure(1, figsize=(8, 6))
		ax = Axes3D(fig, elev=-150, azim=110)
		#X_reduced = PCA(n_components=3).fit_transform(self.data)
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


class DesicionTreeClassifier(BaseClassifier):
	
	def __init__(self,file_path:Path):
		super().__init__(file_path)
		self._initialized = False
	
	def predict_row(self,tree,row):
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

	def predict(self,test):
		self.check_init()
		res=[]
		for idx, row in test.iterrows():
			res.append(self.predict_row(self.tree,row))
		return res

	def get_split(self,X_train:pd.core.frame.DataFrame,y_train:pd.core.series.Series):
		b_index, b_value, b_score, b_groups = 999, 999, 999, []
		for index in range(X_train.shape[1]):
			for idx,row in X_train.iterrows():
				groups = self.get_groups(index, row[index], X_train,y_train)
				gini = self.gini_index(groups)
				if gini < b_score:
					print('here')
					# current b_index = best column for spliting , row[index] - value in this column 
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups
		# absolute best 
		return {'index':b_index, 'value':b_value, 'groups':b_groups}
	
	def get_groups(self,index:int, value:float, X_train:pd.core.frame.DataFrame,y_train:pd.core.series.Series):
		X_left, X_right = pd.DataFrame(),pd.DataFrame()
		for idx,row in X_train.iterrows():
			if row[index] < value:
				X_left=X_left.append([row,],ignore_index=False)
			else:
				X_right=X_right.append(row,ignore_index=False)
		y_left=y_train[X_left.index.values]
		y_right=y_train[X_right.index.values]
		return [X_left,y_left, X_right,y_right]
	
	def gini_index(self,groups:List[pd.core.frame.DataFrame]):
		# count all samples at split point
		n_instances = sum(list(map(len,groups)))/2
		# sum weighted Gini index for each group
		gini = 0
		for i in range(0,len(groups),2):
			size = len(groups[i])
			# avoid divide by zero
			if size == 0:
				continue
			# score the group based on the score for each class
			score = sum(pow(groups[i+1].value_counts() / size,2))
			# weight the group score by its relative size
			gini += (1.0 - score) * (size / n_instances)
		return gini

	def to_terminal(self,group):
		print(group)
		return max(group.value_counts())
	 
	# Create child splits for a node or make terminal
	def split(self,node, depth):
		left,y_left, right,y_right = node['groups']
		del(node['groups'])
		# check for a no split
		if left.empty or right.empty:
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
		print(root)
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

my_models=[GaussianClassifier(f_path),DesicionTreeClassifier(f_path)]
for my_m,m in zip(my_models,models):
	bc=my_m
	stratified_split(VALIDATION_FOLDERS,mode[0],m,bc,bc.X,bc.Y)
	bc.X,bc.Y,real_Y=stratified_split(TEST_FOLDERS,mode[1],m,bc,bc.X,bc.Y)
bc.classes_viz()


'''print('Model {}\nAccuracy:{}\nRecall:{}\nPrecision:{}\n'.format(models[0],
	metrics.accuracy_score(y_test,y_pred),
	metrics.recall_score(y_test,y_pred,average=None),
	metrics.precision_score(y_test,y_pred,average=None)))
'''
