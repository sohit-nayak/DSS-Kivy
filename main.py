__version__ = 1.0

import pandas as pd
import numpy as np

from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.app import App
from kivy.base import runTouchApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
from kivy.properties import ListProperty
from kivy.properties import DictProperty
from kivy.properties import NumericProperty
from kivy.properties import ObjectProperty
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window


				 
from sklearn.svm import SVC # SVM
from sklearn.ensemble import RandomForestClassifier #RandomForest
from sklearn.neighbors import KNeighborsClassifier #KNeighbors
from sklearn.tree import DecisionTreeClassifier #DecisionTree
from sklearn.neural_network import MLPClassifier #ANN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


from scipy import interp
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold




Window.clearcolor = (78/255., 208/255., 155/255., 1)

Builder.load_string('''
<SpinnerOption>:
    size_hint_y: None
    height: 30
''')

from kivy.config import Config
Config.set('graphics', 'fullscreen', 'auto')

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivyagg")
from kivy.garden.matplotlib import FigureCanvasKivy, FigureCanvasKivyAgg

from matplotlib import pyplot as plt
import seaborn as sns


class InternetPopup(Popup):	
	

	def __init__(self, root, **kwargs):
		super(InternetPopup, self).__init__(**kwargs)
		self.root = root
		self.auto_dismiss = True

	def send_file_name(self, value, *args):

		self.root.file_name = self.ids.url.text
		self.dismiss()


class LocalFilePopup(Popup):
	
	def __init__(self, root, **kwargs):
		super(LocalFilePopup, self).__init__(**kwargs)
		self.root = root
		self.auto_dismiss = True


	def select(self, *args):
			self.root.file_name = args[1][0]
			self.dismiss()

class LocalTestFilePopup(Popup):
	test_filename = StringProperty('None')
	

	def __init__(self, root, **kwargs):
			super(LocalTestFilePopup, self).__init__(**kwargs)
			self.root = root
			self.auto_dismiss = True

	def select(self, *args):
		self.root.test_file_name = args[1][0]
		self.test_filename = args[1][0]

	def call_import(self, *args):
		self.root.import_test_dataset()
		self.dismiss()

class GraphPopup(Popup):
	def __init__(self, root, **kwargs):
		super(GraphPopup, self).__init__(**kwargs)
		self.root = root
		self.auto_dismiss = True

	def view_graph(self, *args):
		# # Run classifier with cross-validation and plot ROC curves
		# X = self.root.data.drop(self.root.ids.predict_dropdown_choose_parameter.text, axis=1)
		# y = self.root.data[self.root.ids.predict_dropdown_choose_parameter.text]
		# classes = y.unique()
		# print classes
		# y = label_binarize(y, classes=[0,1,2])
		# n_classes = y.shape[1]

		# # Add noisy features to make the problem harder
		# random_state = np.random.RandomState(0)
		# n_samples, n_features = X.shape
		# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

		# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
		# classifier = OneVsRestClassifier(self.root.model)
		# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
		# # Compute ROC curve and ROC area for each class
		# fpr = dict()
		# tpr = dict()
		# roc_auc = dict()
		# for i in range(n_classes):
		#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		#     roc_auc[i] = auc(fpr[i], tpr[i])


		# # Compute micro-average ROC curve and ROC area
		# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
		# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		# # plt.figure()
		# lw = 2
		# # plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
		# # # plt.plot(fpr[2], tpr[2], color='darkorange',
		# # #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
		# # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		# # plt.xlim([0.0, 1.0])
		# # plt.ylim([0.0, 1.05])
		# # plt.xlabel('False Positive Rate')
		# # plt.ylabel('True Positive Rate')
		# # plt.title('Receiver operating characteristic example')
		# # plt.legend(loc="lower right")
		# # plt.show()
		# # First aggregate all false positive rates
		# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

		# # Then interpolate all ROC curves at this points
		# mean_tpr = np.zeros_like(all_fpr)
		# for i in range(n_classes):
		#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

		# # Finally average it and compute AUC
		# mean_tpr /= n_classes

		# fpr["macro"] = all_fpr
		# tpr["macro"] = mean_tpr
		# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		# # Plot all ROC curves
		# plt.figure()
		# plt.plot(fpr["micro"], tpr["micro"],
		#          label='micro-average ROC curve (area = {0:0.2f})'
		#                ''.format(roc_auc["micro"]),
		#          color='deeppink', linestyle=':', linewidth=4)

		# plt.plot(fpr["macro"], tpr["macro"],
		#          label='macro-average ROC curve (area = {0:0.2f})'
		#                ''.format(roc_auc["macro"]),
		#          color='navy', linestyle=':', linewidth=4)

		# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
		# for i, color in zip(range(n_classes), colors):
		#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		#              label='ROC curve of class {0} (area = {1:0.2f})'
		#              ''.format(i, roc_auc[i]))

		# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		# plt.xlim([0.0, 1.0])
		# plt.ylim([0.0, 1.05])
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title('Some extension of Receiver operating characteristic to multi-class')
		# plt.legend(loc="lower right")
		# # plt.show()

		# # print plt.gcf().axes
		X = self.root.data.drop(self.root.ids.predict_dropdown_choose_parameter.text, axis=1)
		y = self.root.data[self.root.ids.predict_dropdown_choose_parameter.text]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=69)
		self.root.model.fit(X_train, y_train)
		probs = self.root.model.predict_proba(X_test)
		preds = probs[:,1]
		fpr, tpr, threshold = roc_curve(y_test, preds)
		roc_auc = auc(fpr, tpr)

		plt.title('Receiver Operating Characteristic')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		# plt.show()
		print 'done'
		self.ids.graph_area.add_widget(FigureCanvasKivyAgg(plt.gcf()))

class ManualInputPopup(Popup):
	

	def __init__(self, root, **kwargs):
			super(ManualInputPopup, self).__init__(**kwargs)
			self.root = root
			self.auto_dismiss = True

	def populate(self, *args):
		layout = self.ids.layout_manual_test_data

		for col in self.root.column_names:
			layout.add_widget(Label(text=col))
			layout.add_widget(TextInput(multiline=False,size_hint_y=None,height=30))

	def done(self, *args):
		if (self.root.test_data != []):
			self.root.ids.predict_update_status.text = 'Test Data imported successfully! '
		else:
			self.root.ids.predict_update_status.text = 'Test Data empty! Please input data!'
		self.dismiss()


class RootWidget(TabbedPanel):

	file_name = StringProperty('None')
	test_file_name = StringProperty('None')
	big_dict = DictProperty()
	empty_big_dict = DictProperty()
	column_names = []
	update_the_status = StringProperty('Welcome to DSS!')
	number_of_columns = NumericProperty(len(column_names))
	value = NumericProperty()
	columns = ListProperty(column_names)
	data = []
	test_data = []
	checkbox1 = CheckBox(group='algo_selection')
	model = ObjectProperty()
	# X = ''
	# y = ''

	c_lower = ''
	c_upper = ''
	tol_lower = ''
	tol_upper = ''
	degree_lower = ''
	degree_upper = ''
	coef0_lower = ''
	coef0_upper = ''
	n_neighbors_lower = ''
	n_neighbors_upper = ''
	p_lower = ''
	p_upper = ''
	leaf_size_lower = ''
	leaf_size_upper = ''

	params = DictProperty()
	
	def __init__(self, **kwargs):
		super(RootWidget, self).__init__(**kwargs)
		Window.size = (1300, 700)

	def ping(self, *args):
		# self.big_dict[column] = value33333
		print args[0]
		print args[1]
		self.big_dict[args[0]][1] = args[1]
		print self.big_dict[args[0]][0]

	def clean(self, column):
		i = 0
		seperator_list = []
		local_data = self.data[column]
		new_local_data = []
		for entry in local_data:
			if entry:
				if entry not in seperator_list:
					seperator_list.append(entry)
					new_local_data.append(i)
					i += 1
				else :
					new_local_data.append(seperator_list.index(entry))
			else:
				new_local_data.append(0)
		self.data[column] = new_local_data

	def import_dataset(self):

		if self.file_name == 'None':
			self.update_the_status = 'Please import some dataset!'
			return
		if not (self.file_name.endswith('.zip') or self.file_name.endswith('.csv')):
			self.update_the_status = 'Only zip and csv extensions are allowed. More support will be available in future releases!'
			return
		if self.file_name.endswith('zip'):
			# zf = zipfile.ZipFile(self.file_name)333
			self.data = pd.read_csv(self.file_name, compression='zip', sep=',', quotechar='"')
		else:
			self.data = pd.read_csv(self.file_name)
		self.column_names = list(self.data)
		self.number_of_columns = len(self.column_names)
		self.columns = self.column_names
		for val, column in enumerate(self.data.dtypes):
			if column == 'object':
				self.clean(self.data.columns[val])


		self.ids.layout_content.clear_widgets()
		# self.ids.set_features.clear_widgets()
		self.ids.display_info.text = str(self.data.describe())
		self.ids.update_status.text ='Dataset imported successfully!'
		# scroll_layout = self.ids.set_features
		print type(self.data.head())
		
		for sl, column in enumerate(self.column_names):
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			# space = BoxLayout(size_hint=(0.4, 1)) 
			layout = self.ids.layout_content
			name = Label(text=column)
			self.big_dict[checkbox] = [column, False]
			# layout.add_widget(space) 	
			layout.add_widget(checkbox)
			layout.add_widget(name)
		for column in self.column_names:
			lab = Label(text=column, size_hint_x=None, width=100)
			ent = TextInput(size_hint_x=None, width=200)
			# scroll_layout.add_widget(lab)
			# scroll_layout.add_widget(ent)3333

	def import_test_dataset(self):
		if self.test_file_name.endswith('zip'):
			self.test_data = pd.read_csv(self.test_file_name, compression='zip', sep=',', quotechar='"')
		else:
			self.test_data = pd.read_csv(self.test_file_name)

		self.ids.predict_update_status.text = 'Test Data imported successfully! '
		

	def display_drop_section(self):
		for sl, column in enumerate(self.column_names):
			label = Label(text=str(sl+1), size_hint=(0.2,1), pos_hint={'top': 0.5 + self.size_hint[1]/2})
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			space = BoxLayout(size_hint=(0.4, 1))
			print 'done'
			layout = self.ids.layout_content
			name = Label(text=column)
			layout.add_widget(label)
			layout.add_widget(space)
			layout.add_widget(checkbox)
			layout.add_widget(name)

	def draw_graph(self, X, Y, type_graph, hue):

		
		graph_display = self.ids.graph_display
		graph_display.clear_widgets()
		plt.close()
		plt.clf()
		graph = type_graph.text
		# graph_display.add_widget(FigureCanvasKivyAgg(plt.close()))
		sns.set_palette('colorblind')
		
		if graph == 'Count Plot':
			if hue.text == 'Hue' or hue.text == '':
				sns.countplot(data=self.data, x=X.text)
			else:
				sns.countplot(data=self.data, x=X.text, hue=hue.text)
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable. The basic API and options are identical to those for barplot(), so you can compare counts across nested variables.'

		elif graph == 'Pair Plot':
			if hue.text == 'Hue' or hue.text == '':
				sns.pairplot(self.data, size=6, x_vars=X.text, y_vars=Y.text )
			else:
				sns.pairplot(self.data, hue=hue.text, size=6, x_vars=X.text, y_vars=Y.text )
			sns.pairplot(self.data, hue=hue.text, size=6, x_vars=X.text, y_vars=Y.text )
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'By default, this plot will create a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column. The diagonal Axes are treated differently, drawing a plot to show the univariate distribution of the data for the variable in that column.'

		elif graph == 'Factor Plot':
			if hue.text == 'Hue' or hue.text == '':
				sns.factorplot(data=self.data, x=X.text, y=Y.text)
			else:
				sns.factorplot(data=self.data, x=X.text, y=Y.text, col=hue.text)
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'Draws a categorical plot onto a FacetGrid.'

		elif graph == 'Dist Plot':
			g = sns.FacetGrid(self.data, col=hue.text)  
			g.map(sns.distplot, X.text)
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'Flexibly plot a univariate distribution of observations.'

		elif graph == 'Scatter Plot':
			g = sns.FacetGrid(self.data, col=hue.text)  
			g.map(plt.scatter, X.text, Y.text)
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'A normal scatter plot.'

		elif graph == 'Reg Plot':
			g = sns.FacetGrid(self.data, col=hue.text)  
			g.map(sns.regplot, X.text, Y.text)
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'Plots data and a linear regression model fit.'

		elif graph == 'Kde Plot':
			g = sns.FacetGrid(self.data, col=hue.text, row=hue.text)
			g.map(sns.kdeplot, X.text, Y.text)
			plt.xticks(rotation='vertical')
 			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'Fit and plot a univariate or bivariate kernel density estimate.'

		elif graph == 'Joint Plot':
			sns.jointplot(X.text, Y.text, data=self.data, kind='kde')
			plt.xticks(rotation='vertical')
			print plt.gcf().axes
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'Draw a plot of two variables with bivariate and univariate graphs.'

		elif graph == 'Violin Plot':
			sns.violinplot(x=X.text, y=Y.text, data=self.data)
			print plt.gcf().axes
			plt.xticks(rotation='vertical')
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'Draw a combination of boxplot and kernel density estimate.'

		elif graph == 'Bar Plot':
			sns.barplot(x=X.text, y=Y.text, data=self.data)
			print plt.gcf().axes
			plt.xticks(rotation='vertical')
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'A normal Bar plot.'

		elif graph == 'Pie Plot':
			sns.barplot(x=X.text, y=Y.text, data=self.data)
			print plt.gcf().axes
			plt.xticks(rotation='vertical')
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		elif graph == 'Box Plot':
			sns.boxplot(x=X.text, y=Y.text, hue=hue.text, data=self.data);
			print plt.gcf().axes
			plt.xticks(rotation='vertical')
			graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
			self.update_the_status = 'A normal Box plot.'
 
	def predict(self, *args):
		predict_graph_display = self.ids.predict_graph
		sns.set_palette('colorblind')
		sns.countplot(data=self.data, x="survived", hue="pclass")
		print plt.gcf().axes
		predict_graph_display.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		print "done"

	def cross_validate(self, *args):
		classifier = self.ids.choose_classifier.text
		roc_auc = 'Multiclass label'
		precision = 'Multiclass label'
		recall = 'Multiclass label'
		self.prediction()
		print self.model
		self.prediction()
		# self.X = self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1)
		# self.y = self.data[self.ids.predict_dropdown_choose_parameter.text]

		self.model.fit(self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text])
		accuracy = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=5, scoring='accuracy').mean()
		if len(self.data[self.ids.predict_dropdown_choose_parameter.text].unique()) < 3:
			precision = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=5, scoring='average_precision').mean()
		f1_score = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=5, scoring='f1_weighted').mean()
		if len(self.data[self.ids.predict_dropdown_choose_parameter.text].unique()) < 3:
			recall = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=5, scoring='recall').mean()
		if len(self.data[self.ids.predict_dropdown_choose_parameter.text].unique()) < 3:
			roc_auc = cross_val_score(self.model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=5, scoring='roc_auc').mean()
		self.ids.accuracy.text = str(accuracy)
		self.ids.precision.text = str(precision)
		self.ids.f1.text = str(f1_score)
		self.ids.recall.text = str(recall)
		self.ids.auc_roc.text = str(roc_auc)

	def prediction(self):
		classifier_type = self.ids.choose_classifier.text
		params = self.params
		print params
		if classifier_type == 'SVM':
			C = float(params['C'])
			kernel = params['kernel']
			degree = int(params['degree'])
			# gamma = float(params['gamma'])
			tol = float(params['tol'])
			coef0 = float(params['coef0'])

			self.model = SVC(C=C, kernel=kernel, degree=degree, tol=tol, coef0=coef0, probability=True)

		if classifier_type == 'ANN':
			hidden_layer_sizes = tuple(params['hidden_layer_sizes'])
			max_iter = int(params['max_iter'])
			# activation = params['activation']
			solver = params['solver']
			learning_rate = params['learning_rate']
			# momentum = params['momentum']

			self.model = MLPClassifier()

		if classifier_type == 'Random Forest':
			n_estimators = int(params['n_estimators'])
			min_samples_leaf = float(params['min_samples_leaf'])
			# max_depth = int(params['max_depth'])
			min_samples_split = float(params['min_samples_split'])
			min_weight_fraction_leaf = float(params['min_weight_fraction_leaf'])
			# max_leaf_nodes = int(params['max_leaf_nodes'])

			self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf)

		if classifier_type == 'Decision Tree':
			max_depth_DT = int(params['max depth'])
			min_samples_split_DT = int(params['min samples split'])
			min_samples_leaf_DT = int(params['min samples leaf'])
			min_weight_fraction_leaf_DT = float(params['min weight fraction leaf'])
			max_leaf_nodes_DT = int(params['max leaf nodes'])
			presort = False
			if params['presort'] == 'True':
				presort = True


			self.model = DecisionTreeClassifier(max_depth=max_depth_DT, min_samples_split=min_samples_split_DT, min_samples_leaf=min_samples_leaf_DT,
			min_weight_fraction_leaf=min_weight_fraction_leaf_DT, max_leaf_nodes=max_leaf_nodes_DT, presort=presort)

		if classifier_type == 'KNN': 
			n_neighbors = int(params['n_neighbors'])
			# p = int(params['p'])
			leaf_size = int(params['leaf_size'])
			n_jobs = int(params['n_jobs'])
			algorithm = params['algorithm']
			weights = params['weights']

			self.model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, n_jobs=n_jobs, algorithm=algorithm, weights=weights)
		print self.model
		print classifier_type



	def drop_columns(self, *args):

		self.ids.update_status.text ='Columns dropped successfully!'
		for checkbox in self.big_dict:
			if self.big_dict[checkbox][1]:
				self.data.drop(self.big_dict[checkbox][0], axis=1)
				self.column_names.remove(self.big_dict[checkbox][0])
		# self.display_drop_section()
		self.ids.layout_content.clear_widgets()
		# self.ids.set_features.clear_widgets()
		self.number_of_columns = len(self.column_names)
		self.big_dict = self.empty_big_dict
		for sl, column in enumerate(self.column_names):
			checkbox = CheckBox(text=column)
			checkbox.bind(active=self.ping)
			self.big_dict[checkbox] = [column, False]
			layout = self.ids.layout_content
			name = Label(text=column)
			layout.add_widget(checkbox)
			layout.add_widget(name)

		for column in self.column_names:
			lab = Label(text=column, size_hint_x=None, width=100)
			ent = TextInput(size_hint_x=None, width=200)
			# self.ids.set_features.add_widget(lab)
			# self.ids.set_features.add_widget(ent)

		self.column_name = self.column_names
		self.columns = self.column_names
		self.ids.display_info.text = str(self.data.describe())

	def dropDown(self, lists, *args):
		dropdown = DropDown()
		for names in lists:
			btn = Button(text=names, size_hint_y=None, height=30)
			btn.bind(on_release=lambda btn: dropdown.select(btn.text))
			dropdown.add_widget(btn)
		args[0].bind(on_release=dropdown.open)
		dropdown.bind(on_select=lambda instance, x: setattr(args[0] ,'text', x))
		# scroll = ScrollView(size_hint=(1, None), do_scroll_y=True, do_scroll_x=False)
		# scroll.add_widget(self.ids.layout_dropdown)

	def optimize_algo_selection(self, *args):
		check = self.ids.predict_checkbox_algo
		layout = self.ids.predict_optimize_algo_selection
		if check.active == True:
			layout.clear_widgets()

			
			self.checkbox1.bind(on_release=self.populate_predict_algo_parameters, on_press=lambda x: self.optimize_data('geneticalgo'))
			label1 = Label(text='GridSearchCV')
			checkbox2 = CheckBox(group='algo_selection')
			checkbox2.bind( on_press=lambda x: self.optimize_data('geneticalgo'))
			label2 = Label(text='Genetic Algorithm')
			layout.add_widget(self.checkbox1)		
			layout.add_widget(label1)
			layout.add_widget(checkbox2)
			layout.add_widget(label2)
		else:
			layout.clear_widgets()

	def populate_predict_algo_parameters(self, *args):
		layout = self.ids.predict_optimize_algo_parameters
		limits = {}

		if self.checkbox1.active == True:
			if self.ids.choose_classifier.text == 'SVM':
				limits = {}
				limits['type'] = 'SVM'
				layout.clear_widgets()
				c_label = Label(text='c', color=(1,1,1,2))
				self.c_lower = TextInput(text='0.0', multiline=False,
					                                   size_hint=(None, None), height=30,width=140, hint_text='lower c value')
				self.c_upper = TextInput(text='0.0', multiline=False,
					                                   size_hint=(None, None), height=30,width=140, hint_text='upper c value')
				layout.add_widget(c_label)
				layout.add_widget(self.c_lower)
				layout.add_widget(self.c_upper)
				# c_lo = NumericProperty(float(c_lower.text))
				# c_hi = NumericProperty(float(c_upper.text))
				# limits['C'] = [c_lo, c_hi]
				# self.c_lo = float(c_lower.text)
				# self.c_hi = float(c_upper.text)

				tol_label = Label(text='tol', color=(1,1,1,2))
				self.tol_lower = TextInput(text='0.0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower tol value')
				self.tol_upper = TextInput(text='0.0',multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper tol value')
				layout.add_widget(tol_label)
				layout.add_widget(self.tol_lower)
				layout.add_widget(self.tol_upper)
				# limits['tol'] = [tol_lower, tol_upper]
				# self.tol_lo = float(tol_lower.text)
				# self.tol_hi = float(tol_upper.text)

				degree_label = Label(text='degree', color=(1,1,1,2))
				self.degree_lower = TextInput(text='0',multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower degree value')
				self.degree_upper = TextInput(text='0',multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper degree value')
				layout.add_widget(degree_label)
				layout.add_widget(self.degree_lower)
				layout.add_widget(self.degree_upper)
				# limits['degree'] = [degree_lower, degree_upper]3
				# self.degree_lo = int(degree_lower.text)
				# self.degree_hi = int(degree_upper.text)

				gamma_label = Label(text='gamma', color=(1,1,1,2))
				gamma_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower gamma value')
				gamma_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper gamma value')
				layout.add_widget(gamma_label)
				layout.add_widget(gamma_lower)
				layout.add_widget(gamma_upper)
				limits['gamma'] = [gamma_lower, gamma_upper]

				coef0_label = Label(text='coef0', color=(1,1,1,2))
				self.coef0_lower = TextInput(text='0.0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower coef0 value')
				self.coef0_upper = TextInput(text='0.0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper coef0 value')
				layout.add_widget(coef0_label)
				layout.add_widget(self.coef0_lower)
				layout.add_widget(self.coef0_upper)
				# limits['coef0'] = [coef0_lower, coef0_upper]333
				# self.coef0_lo = float(coef0_lower.text)
				# self.coef0_hi = float(coef0_upper.text)

				kernel_label = Label(text = 'kernel', color=(1,1,1,2))
				kernel_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
				kernel_mainbutton.bind(on_press=lambda x:self.dropDown(['rbf','linear','poly','sigmoid','precomputed'], kernel_mainbutton))
				layout.add_widget(kernel_label)
				layout.add_widget(kernel_mainbutton)
				return limits

			if self.ids.choose_classifier.text == 'Random Forest':
				layout.clear_widgets()
				limits = {}
				limits['type'] = 'Random Forest'
				n_estimators_label = Label(text='n estimators', color=(1,1,1,2))
				self.n_estimators_lower = TextInput(tex3t='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				self.n_estimators_upper = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(n_estimators_label)
				layout.add_widget(self.n_estimators_lower)
				layout.add_widget(self.n_estimators_upper)
				# limits['n_estimators'] = [n_estimators_lower, n_estimators_upper]

				min_samples_leaf_label = Label(text='min samples\n    leaf', color=(1,1,1,2))
				self.min_samples_leaf_lower = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				self.min_samples_leaf_upper = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_samples_leaf_label)
				layout.add_widget(self.min_samples_leaf_lower)
				layout.add_widget(self.min_samples_leaf_upper)
				# limits['min_sample_leaf'] = [min_samples_leaf_lower, min_samples_leaf_upper]

				max_depth_label = Label(text='max depth', color=(1,1,1,2))
				max_depth_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_depth_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_depth_label)
				layout.add_widget(max_depth_lower)
				layout.add_widget(max_depth_upper)
				limits['max_depth'] = [max_depth_lower, max_depth_upper]

				min_samples_split_label = Label(text='min samples\n     split', color=(1,1,1,2))
				self.min_samples_split_lower = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				self.min_samples_split_upper = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_samples_split_label)
				layout.add_widget(self.min_samples_split_lower)
				layout.add_widget(self.min_samples_split_upper)
				# limits['min_samples_split'] = [min_samples_split_lower, min_samples_split_upper]

				min_weight_fraction_leaf_label = Label(text='min weight\nfraction leaf', color=(1,1,1,2))
				min_weight_fraction_leaf_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				min_weight_fraction_leaf_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_weight_fraction_leaf_label)
				layout.add_widget(min_weight_fraction_leaf_lower)
				layout.add_widget(min_weight_fraction_leaf_upper)
				limits['min_weight_fraction_leaf'] = [min_weight_fraction_leaf_lower, min_weight_fraction_leaf_upper]

				max_leaf_nodes_label = Label(text='max leaf\n  nodes', color=(1,1,1,2))
				max_leaf_nodes_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_leaf_nodes_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_leaf_nodes_label)
				layout.add_widget(max_leaf_nodes_lower)
				layout.add_widget(max_leaf_nodes_upper)
				limits['max_leaf_nodes'] = [max_leaf_nodes_lower, max_leaf_nodes_upper]
				return limits

			if self.ids.choose_classifier.text == 'Decision Tree':
				layout.clear_widgets()
				max_depth_DT_label = Label(text='max depth', color=(1,1,1,2))
				max_depth_DT_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_depth_DT_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_depth_DT_label)
				layout.add_widget(max_depth_DT_lower)
				layout.add_widget(max_depth_DT_upper)

				min_samples_split_DT_label = Label(text='min samples split', color=(1,1,1,2))
				min_samples_split_DT_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				min_samples_split_DT_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_samples_split_DT_label)
				layout.add_widget(min_samples_split_DT_lower)
				layout.add_widget(min_samples_split_DT_upper)

				min_samples_leaf_DT_label = Label(text='min samples leaf_size', color=(1,1,1,2))
				min_samples_leaf_DT_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				min_samples_leaf_DT_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_samples_leaf_DT_label)
				layout.add_widget(min_samples_leaf_DT_lower)
				layout.add_widget(min_samples_leaf_DT_upper)

				min_weight_fraction_leaf_DT_label = Label(text='min weight \n fraction leaf', color=(1,1,1,2))
				min_weight_fraction_leaf_DT_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				min_weight_fraction_leaf_DT_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(min_weight_fraction_leaf_DT_label)
				layout.add_widget(min_weight_fraction_leaf_DT_lower)
				layout.add_widget(min_weight_fraction_leaf_DT_upper)

				max_leaf_nodes_DT_label = Label(text='max leaf nodes', color=(1,1,1,2))
				max_leaf_nodes_DT_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_leaf_nodes_DT_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_leaf_nodes_DT_label)
				layout.add_widget(max_leaf_nodes_DT_lower)
				layout.add_widget(max_leaf_nodes_DT_upper)

				presort_label = Label(text = 'presort', color=(1,1,1,2))
				presort_mainbutton = Button(text='Choose presort',size_hint=(None, None), height=30,width=140)
				presort_mainbutton.bind(on_press=lambda x:self.dropDown(['False', 'True'], presort_mainbutton))
				layout.add_widget(presort_label)
				layout.add_widget(presort_mainbutton)	


			if self.ids.choose_classifier.text == 'KNN':
				layout.clear_widgets()
				limits = {}
				# limits['type'] = 'KNN'
				n_neighbors_label = Label(text='n neighbors', color=(1,1,1,2))
				self.n_neighbors_lower = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				self.n_neighbors_upper = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(n_neighbors_label)
				layout.add_widget(self.n_neighbors_lower)
				layout.add_widget(self.n_neighbors_upper)
				# limits['n_neighbors'] = [self.n_neighbors_lower, n_neighbors_upper]

				p_label = Label(text='p', color=(1,1,1,2))
				self.p_lower = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower p value')
				self.p_upper = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper p value')
				layout.add_widget(p_label)
				layout.add_widget(self.p_lower)
				layout.add_widget(self.p_upper)
				# limits['p'] = [p_lower, p_upper]

				leaf_size_label = Label(text='leaf size', color=(1,1,1,2))
				self.leaf_size_lower = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				self.leaf_size_upper = TextInput(text='0', multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(leaf_size_label)
				layout.add_widget(self.leaf_size_lower)
				layout.add_widget(self.leaf_size_upper)
				# limits['leaf_size'] = [leaf_size_lower, leaf_size_upper]

				n_jobs_label = Label(text='n jobs', color=(1,1,1,2))
				n_jobs_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				n_jobs_upper = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(n_jobs_label)
				layout.add_widget(n_jobs_lower)
				layout.add_widget(n_jobs_upper)
				limits['n_jobs'] = [n_jobs_lower, n_jobs_upper]

				algorithm_label = Label(text = 'algorithm', color=(1,1,1,2))
				algorithm_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
				algorithm_mainbutton.bind(on_press=lambda x:self.dropDown(['ball tree','kd tree','brute','auto'], algorithm_mainbutton))
				layout.add_widget(algorithm_label)
				layout.add_widget(algorithm_mainbutton)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)

				weights_label = Label(text = 'weights', color=(1,1,1,2))
				weights_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
				weights_mainbutton.bind(on_press=lambda x:self.dropDown(['rbf','uniform', 'distance'], weights_mainbutton))
				layout.add_widget(weights_label)
				layout.add_widget(weights_mainbutton)
				return limits

			if self.ids.choose_classifier.text == 'ANN':
				layout.clear_widgets()
				limits = {}
				limits['type'] = 'ANN'
				hidden_layer_sizes_label = Label(text='hidden layer\n   sizes', color=(1,1,1,2))
				hidden_layer_sizes_input_lower= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				hidden_layer_sizes_input_upper= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(hidden_layer_sizes_label)
				layout.add_widget(hidden_layer_sizes_input_lower)
				layout.add_widget(hidden_layer_sizes_input_upper)
				limits['hidden_layer_sizes'] = [hidden_layer_sizes_input_lower, hidden_layer_sizes_input_upper]


				max_iter_label = Label(text='max iter', color=(1,1,1,2))
				max_iter_input_lower = TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				max_iter_input_upper= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(max_iter_label)
				layout.add_widget(max_iter_input_lower)
				layout.add_widget(max_iter_input_upper)
				limits['max_iter'] = [max_iter_input_lower, max_iter_input_upper]


				activation_label = Label(text = 'activation', color=(1,1,1,2))
				activation_spinner = Spinner(text = 'Select', values=['identity', 'logistic', 'tanh', 'relu'])
				layout.add_widget(activation_label)
				layout.add_widget(activation_spinner)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)

				solver_label = Label(text = 'solver', color=(1,1,1,2))
				solver_spinner = Spinner(text = 'Select', values=['lbfgs', 'sgd', 'adam'])
				layout.add_widget(solver_label)
				layout.add_widget(solver_spinner)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)

				learning_rate_label = Label(text = 'learning rate', color=(1,1,1,2))
				learning_rate_spinner = Spinner(text = 'Select', values= ['adaptive', 'invscaling','constant'])
				layout.add_widget(learning_rate_label)
				layout.add_widget(learning_rate_spinner)

				dummy_label = Label(size_hint=(None, None), height=30,width=140)
				layout.add_widget(dummy_label)


				# if solver_spinner.text == 'sgd':
				momentum_label = Label(text='momentum', color=(1,1,1,2))
				momentum_input_lower= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='lower value')
				momentum_input_upper= TextInput(multiline=False,
	                                   size_hint=(None, None), height=30,width=140, hint_text='upper value')
				layout.add_widget(momentum_label)
				layout.add_widget(momentum_input_lower)
				layout.add_widget(momentum_input_upper)
				limits['momentum'] = [momentum_input_lower, momentum_input_upper]

				return limits
		else:
			layout.clear_widgets()

	def grid_search(self):

		model_type = self.ids.choose_classifier.text
		parameters = {}
		scoring= []
		scores = ['accuracy', 'average_precision', 'f1_weighted', 'recall', 'roc_auc']

		if model_type == 'SVM':
			# collect and arrange data
			print self.c_lower
			if self.c_lower.text != '0.0' and self.c_upper.text != '0.0':
				c_lo = float(self.c_lower.text)
				c_hi = float(self.c_upper.text)
				c_range = np.arange(c_lo, c_hi, 0.1)
				parameters['C'] = c_range
				print 'gotcha'

			if self.tol_lower.text != '0.0' and self.tol_upper.text != '0.0':
				tol_lo = float(self.tol_lower.text)
				tol_hi = float(self.tol_upper.text)
				tol_range = np.arange(tol_lo, tol_hi, 0.01)
				parameters['tol'] = tol_range

			if self.degree_lower.text != '0' and self.degree_upper.text != '0':
				degree_lo = int(self.degree_lower.text)
				degree_hi = int(self.degree_upper.text)
				degree_range = np.arange(degree_lo, degree_hi, 1)
				parameters['degree'] = degree_range

			if self.coef0_lower.text != '0.0' and self.coef0_upper.text != '0.0':
				coef0_lo = float(self.coef0_lower.text)
				coef0_hi = float(self.coef0_upper.text)
				coef0_range = np.arange(coef0_lo, coef0_hi, 0.1)
				parameters['coef0'] = coef0_range
			for score in scores:

				model = SVC()
				cv_model = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='%s' % score)
				cv_model.fit(self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text])
				print cv_model.best_params_, 'done'
				print cv_model.best_score_
				print cv_model.cv_results_
				scoring.append(cv_model.best_score_)

			self.ids.accuracy.text = str(scoring[0])
			self.ids.precision.text = str(scoring[1])
			self.ids.f1.text = str(scoring[2])
			self.ids.recall.text = str(scoring[3])
			self.ids.auc_roc.text = str(scoring[4])
				

		if model_type == 'KNN':
			# collect and arrange data
			if self.n_neighbors_lower.text != '0' and self.n_neighbors_upper.text != '0':
				n_neighbors_lo = int(self.n_neighbors_lower.text)
				n_neighbors_hi = int(self.n_neighbors_upper.text)
				n_neighbors_range = np.arange(n_neighbors_lo, n_neighbors_hi, 1)
				print n_neighbors_range
				parameters['n_neighbors'] = n_neighbors_range

			if self.p_lower.text != '0' and self.p_upper.text != '0':
				p_lo = int(self.p_lower.text)
				p_hi = int(self.p_upper.text)
				p_range = np.arange(p_lo, p_hi, 1)
				parameters['p'] = p_range

			if self.leaf_size_lower.text != '0' and self.leaf_size_upper.text != '0':
				leaf_size_lo = int(self.leaf_size_lower.text)
				leaf_size_hi = int(self.leaf_size_upper.text)
				leaf_size_range = np.arange(leaf_size_lo, leaf_size_hi, 1)
				parameters['leaf_size'] = leaf_size_range

			for score in scores:
				model = KNeighborsClassifier()
				cv_model = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='%s' % score)
				cv_model.fit(self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text])
				print cv_model.best_params_, 'done'
				print cv_model.best_score_
				print cv_model.cv_results_
				scoring.append(cv_model.best_score_)

			self.ids.accuracy.text = str(scoring[0])
			self.ids.precision.text = str(scoring[1])
			self.ids.f1.text = str(scoring[2])
			self.ids.recall.text = str(scoring[3])
			self.ids.auc_roc.text = str(scoring[4])

		if model_type == 'Random Forest':
			# collect and arrange data
			if self.n_estimators_lower.text != '0' and self.n_estimators_upper.text != '0':
				n_estimators_lo = int(self.n_estimators_lower.text)
				n_estimators_hi = int(self.n_estimators_upper.text)
				n_estimators_range = np.arange(n_estimators_lo, n_estimators_hi, 1)
				print n_estimators_range
				parameters['n_estimators'] = n_estimators_range

			if self.min_samples_leaf_lower.text != '0' and self.min_samples_leaf_split_upper.text != '0':
				min_samples_leaf_lo = int(self.min_samples_leaf_lower.text)
				min_samples_leaf_hi = int(self.min_samples_leaf_upper.text)
				min_samples_leaf_range = np.arange(min_samples_leaf_lo, min_samples_leaf_hi, 1)
				parameters['min_samples_leaf'] = min_samples_leaf_range

			if self.min_samples_split_lower.text != '0' and self.min_samples_split_upper.text != '0':
				min_samples_split_lo = int(self.min_samples_split_lower.text)
				min_samples_split_hi = int(self.min_samples_split_upper.text)
				min_samples_split_range = np.arange(min_samples_split_lo, min_samples_split_hi, 1)
				parameters['min_samples_split'] = min_samples_split_range

			for score in scores:
				model = RandomForestClassifier()
				cv_model = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='%s' % score)
				cv_model.fit(self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text])
				print 'best parameters: ', cv_model.best_params_
				print cv_model.best_score_
				scoring.append(cv_model.best_score_)

			self.ids.accuracy.text = str(scoring[0])
			self.ids.precision.text = str(scoring[1])
			self.ids.f1.text = str(scoring[2])
			self.ids.recall.text = str(scoring[3])
			self.ids.auc_roc.text = str(scoring[4])

		# self.ids.accuracy.text = str(scoring[0])
		# self.ids.precision.text = str(scoring[1])
		# self.ids.f1.text = str(scoring[2])
		# self.ids.recall.text = str(scoring[3])
		# self.ids.auc_roc.text = str(scoring[4])

	def manual_parameter_selection(self, *args):
		pass

	def optimize_data(self, name):
		
		classifier = self.ids.choose_classifier.text
		model = self.prediction()

		if name == 'gridcv':
			pass
			# accuracy = cross_val_score(model, sellf.f.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='accuracy').mean()
			# precision = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='average_precision').mean()
			# f1_score = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='f1').mean()
			# recall = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='recall').mean()
			# roc_auc = cross_val_score(model, self.data.drop(self.ids.predict_dropdown_choose_parameter.text, axis=1), self.data[self.ids.predict_dropdown_choose_parameter.text], cv=10, scoring='roc_auc').mean()
			# self.ids.accuracy.text = accuracy
			# self.ids.precision.text = precision
			# self.ids.f1.text = f1_score
			# self.ids.recall.text = recall
			# self.ids.auc_roc.text = roc_auc

	def updateSubSpinner(self, text):
		self.ids.choose_classifier.text = '< Select >'

		if text == 'Tree based':
			self.ids.choose_classifier.values = ['Random Forest','Decision Tree']
		if text == 'Non-Tree based':
			self.ids.choose_classifier.values = ['SVM','ANN','KNN']

	def internet_popup(self, *args):
		internet = InternetPopup(self)
		internet.open()

	def local_file_popup(self, *args):
		local = LocalFilePopup(self)
		local.open()

	def local_test_file_popup(self, *args):
		local_test = LocalTestFilePopup(self)
		local_test.open()

	def test_data_popup(self, *args):
		test = TestDataPopup(self)
		test.open()

	def manual_input_popup(self, *args):
		man_input = ManualInputPopup(self)
		man_input.open()

	def graph_popup(self, *args):
		graph_input = GraphPopup(self)
		graph_input.open()

	def predict_model_parameters(self, value):
		layout = self.ids.layout_predict_parameters

		if value=='SVM':


			layout.clear_widgets()
			c_label = Label(text='C', color=(1,1,1,2))
			c_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='1.0')
			layout.add_widget(c_label)
			layout.add_widget(c_input)
			self.params['C'] = c_input.text

			tol_label = Label(text='tol', color=(1,1,1,2))
			tol_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.049787')
			layout.add_widget(tol_label)
			layout.add_widget(tol_input)
			self.params['tol'] = tol_input.text

			degree_label = Label(text='degree', color=(1,1,1,2))
			degree_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='3')
			layout.add_widget(degree_label)
			layout.add_widget(degree_input)
			self.params['degree'] = degree_input.text

			gamma_label = Label(text='gamma', color=(1,1,1,2))
			gamma_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='auto')
			layout.add_widget(gamma_label)
			layout.add_widget(gamma_input)
			self.params['gamma'] = gamma_input.text

			coef0_label = Label(text='coef0', color=(1,1,1,2))
			coef0_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.0')
			layout.add_widget(coef0_label)
			layout.add_widget(coef0_input)
			self.params['coef0'] = coef0_input.text

			kernel_label = Label(text = 'kernel', color=(1,1,1,2))
			kernel_spinner = Spinner(text='rbf',values=['linear','poly','sigmoid','precomputed','rbf'])
			layout.add_widget(kernel_label)
			layout.add_widget(kernel_spinner)
			self.params['kernel'] = kernel_spinner.text

			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()

		if value=='Random Forest' :

			layout.clear_widgets()
			n_estimators_label = Label(text='n estimators', color=(1,1,1,2),size=self.parent.size)
			n_estimators_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='10')
			layout.add_widget(n_estimators_label)
			layout.add_widget(n_estimators_input)
			self.params['n_estimators'] = n_estimators_input.text

			min_samples_leaf_label = Label(text='min samples\n    leaf', color=(1,1,1,2),size=self.parent.size)
			min_samples_leaf_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.5')
			layout.add_widget(min_samples_leaf_label)
			layout.add_widget(min_samples_leaf_input)
			self.params['min_samples_leaf'] = min_samples_leaf_input.text


			max_depth_label = Label(text='max depth', color=(1,1,1,2),size=self.parent.size)
			max_depth_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='None')
			layout.add_widget(max_depth_label)
			layout.add_widget(max_depth_input)
			self.params['max_depth'] = max_depth_input.text

			min_samples_split_label = Label(text='min samples\n     split', color=(1,1,1,2),size=self.parent.size)
			min_samples_split_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='2')
			layout.add_widget(min_samples_split_label)
			layout.add_widget(min_samples_split_input)
			self.params['min_samples_split'] = min_samples_split_input.text

			min_weight_fraction_leaf_label = Label(text='min weight\nfraction leaf', color=(1,1,1,2),size=self.parent.size)
			min_weight_fraction_leaf_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.0')
			layout.add_widget(min_weight_fraction_leaf_label)
			layout.add_widget(min_weight_fraction_leaf_input)
			self.params['min_weight_fraction_leaf'] = min_weight_fraction_leaf_input.text

			max_leaf_nodes_label = Label(text='max leaf\n  nodes', color=(1,1,1,2),size=self.parent.size)
			max_leaf_nodes_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='None')
			layout.add_widget(max_leaf_nodes_label)
			layout.add_widget(max_leaf_nodes_input)
			self.params['max_leaf_nodes'] = max_leaf_nodes_input.text

			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()

		if value=='Decision Tree':
			layout.clear_widgets()
			max_depth_DT_label = Label(text='max depth', color=(1,1,1,2))
			max_depth_DT_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='1')
			layout.add_widget(max_depth_DT_label)
			layout.add_widget(max_depth_DT_input)
			self.params['max depth'] = max_depth_DT_input.text

			min_samples_split_DT_label = Label(text='min samples split', color=(1,1,1,2))
			min_samples_split_DT_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='2')
			layout.add_widget(min_samples_split_DT_label)
			layout.add_widget(min_samples_split_DT_input)
			self.params['min samples split'] = min_samples_split_DT_input.text

			min_samples_leaf_DT_label = Label(text='min samples leaf', color=(1,1,1,2))
			min_samples_leaf_DT_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='1')
			layout.add_widget(min_samples_leaf_DT_label)
			layout.add_widget(min_samples_leaf_DT_input)
			self.params['min samples leaf'] = min_samples_leaf_DT_input.text

			min_weight_fraction_leaf_DT_label = Label(text='min weight_fraction leaf', color=(1,1,1,2))
			min_weight_fraction_leaf_DT_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.')
			layout.add_widget(min_weight_fraction_leaf_DT_label)
			layout.add_widget(min_weight_fraction_leaf_DT_input)
			self.params['min weight fraction leaf'] = min_weight_fraction_leaf_DT_input.text

			max_leaf_nodes_DT_label = Label(text='max leaf nodes', color=(1,1,1,2))
			max_leaf_nodes_DT_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='2')
			layout.add_widget(max_leaf_nodes_DT_label)
			layout.add_widget(max_leaf_nodes_DT_input)
			self.params['max leaf nodes'] = max_leaf_nodes_DT_input.text

			presort_label = Label(text = 'presort', color=(1,1,1,2))
			presort_spinner = Spinner(text='False', values=['True','False'])
			layout.add_widget(presort_label)
			layout.add_widget(presort_spinner)
			self.params['presort'] = presort_spinner.text

			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()


		if value=='KNN':
			layout.clear_widgets()
			n_neighbors_label = Label(text='n neighbors', color=(1,1,1,2))
			n_neighbors_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='5')
			layout.add_widget(n_neighbors_label)
			layout.add_widget(n_neighbors_input)
			self.params['n_neighbors'] = n_neighbors_input.text

			p_label = Label(text='p', color=(1,1,1,2))
			p_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='0.0')
			layout.add_widget(p_label)
			layout.add_widget(p_input)
			self.params['p'] = p_input.text

			leaf_size_label = Label(text='leaf size', color=(1,1,1,2))
			leaf_size_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='30')
			layout.add_widget(leaf_size_label)
			layout.add_widget(leaf_size_input)
			self.params['leaf_size'] = leaf_size_input.text

			n_jobs_label = Label(text='n jobs', color=(1,1,1,2))
			n_jobs_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='1')
			layout.add_widget(n_jobs_label)
			layout.add_widget(n_jobs_input)
			self.params['n_jobs'] = n_jobs_input.text

			algorithm_label = Label(text = 'algorithm', color=(1,1,1,2))
			algorithm_spinner = Spinner(text='auto',values=['ball tree','kd tree','brute','auto'])
			layout.add_widget(algorithm_label)
			layout.add_widget(algorithm_spinner)
			self.params['algorithm'] = algorithm_spinner.text


			weights_label = Label(text = 'weights', color=(1,1,1,2))
			weights_spinner = Spinner(text='uniform', values=['rbf', 'distance','uniform'])
			layout.add_widget(weights_label)
			layout.add_widget(weights_spinner)
			self.params['weights'] = weights_spinner.text


			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()

		if value == 'ANN':
			layout.clear_widgets()
			hidden_layer_sizes_label = Label(text='hidden layer\n   sizes', color=(1,1,1,2))
			hidden_layer_sizes_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='100')
			layout.add_widget(hidden_layer_sizes_label)
			layout.add_widget(hidden_layer_sizes_input)
			self.params['hidden_layer_sizes'] = hidden_layer_sizes_input.text


			max_iter_label = Label(text='max iter', color=(1,1,1,2))
			max_iter_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='200')
			layout.add_widget(max_iter_label)
			layout.add_widget(max_iter_input)
			self.params['max_iter'] = max_iter_input.text


			activation_label = Label(text = 'activation', color=(1,1,1,2))
			activation_spinner = Spinner(text = 'relu', values=['identity', 'logistic', 'tanh', 'relu'])
			layout.add_widget(activation_label)
			layout.add_widget(activation_spinner)
			self.params['n_estimators'] = activation_spinner.text


			solver_label = Label(text = 'solver', color=(1,1,1,2))
			solver_spinner = Spinner(text = 'adam', values=['lbfgs', 'sgd', 'adam'])
			layout.add_widget(solver_label)
			layout.add_widget(solver_spinner)
			self.params['solver'] = solver_spinner.text



			learning_rate_label = Label(text = 'learning rate', color=(1,1,1,2))
			learning_rate_spinner = Spinner(text = 'constant', values= ['adaptive', 'invscaling','constant'])
			layout.add_widget(learning_rate_label)
			layout.add_widget(learning_rate_spinner)
			self.params['learning_rate'] = learning_rate_spinner.text



			# if solver_spinner.text == 'sgd':
			momentum_label = Label(text='momentum', color=(1,1,1,2))
			momentum_input = TextInput(multiline=False,
                                   size_hint=(None, None), height=30,width=140, text='.9')
			layout.add_widget(momentum_label)
			layout.add_widget(momentum_input)
			
			if self.checkbox1.active == True:
				self.populate_predict_algo_parameters()
				

	# def predict_model_parameters(self, value):
	# 	layout = self.ids.layout_optimize_parameters

	# 	if value=='SVM':
	# 		layout.clear_widgets()
	# 		c_label = Label(text='C', color=(1,1,1,2))
	# 		c_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower c value')
	# 		c_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper c value')
	# 		layout.add_widget(c_label)
	# 		layout.add_widget(c_lower)
	# 		layout.add_widget(c_upper)

	# 		tol_label = Label(text='tol', color=(1,1,1,2))
	# 		tol_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower tol value')
	# 		tol_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper tol value')
	# 		layout.add_widget(tol_label)
	# 		layout.add_widget(tol_lower)
	# 		layout.add_widget(tol_upper)

	# 		degree_label = Label(text='degree', color=(1,1,1,2))
	# 		degree_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower degree value')
	# 		degree_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper degree value')
	# 		layout.add_widget(degree_label)
	# 		layout.add_widget(degree_lower)
	# 		layout.add_widget(degree_upper)

	# 		gamma_label = Label(text='gamma', color=(1,1,1,2))
	# 		gamma_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower gamma value')
	# 		gamma_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper gamma value')
	# 		layout.add_widget(gamma_label)
	# 		layout.add_widget(gamma_lower)
	# 		layout.add_widget(gamma_upper)

	# 		coef0_label = Label(text='coef0', color=(1,1,1,2))
	# 		coef0_lower = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='lower coef0 value')
	# 		coef0_upper = TextInput(multiline=False,
 #                                   size_hint=(None, None), height=30,width=140, hint_text='upper coef0 value')
	# 		layout.add_widget(coef0_label)
	# 		layout.add_widget(coef0_lower)
	# 		layout.add_widget(coef0_upper)

	# 		kernel_label = Label(text = 'kernel', color=(1,1,1,2))
	# 		kernel_mainbutton = Button(text='Choose kernel',size_hint=(None, None), height=30,width=140)
	# 		kernel_mainbutton.bind(on_press=lambda x:self.dropDown(['rbf','linear','poly','sigmoid','precomputed'], kernel_mainbutton))
	# 		layout.add_widget(kernel_label)
	# 		layout.add_widget(kernel_mainbutton)

	# 	if value==2 :
	# 		layout.clear_widgets()
	# 		

	# 	if value==3:
	# 		layout.clear_widgets()


		return self.params

				



class DssApp(App):
    def build(self):
        return RootWidget()


if __name__ == '__main__':
    DssApp().run()