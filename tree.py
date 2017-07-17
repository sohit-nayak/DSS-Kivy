from sklearn.externals.six import StringIO
import pydotplus
from sklearn.datasets import load_iris
import numpy as numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import pydot

#viz code
from sklearn.externals.six import StringIO
import pydot
import pandas as pd

# data = load_iris()
data = pd.read_csv('clean_titanic.csv')
X = data.drop(['survived'], axis=1)
y = data.survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=6)
forest = DecisionTreeClassifier()
forest.fit(X_train, y_train)

#viz code
dot_data = StringIO()
# print len(forest.estimators_)
i_tree = 0
export_graphviz(forest, out_file=dot_data,
                            # feature_names=data.feature_names,
                            # class_names=data.target_names,
                            filled=True,
                            rounded=True,
                            impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")
graph.write_png('str(estimator)' + '.png')
# for tree_in_forest in forest.estimators_:
#     with open('tree_' + str(i_tree) + '.png', 'w') as my_file:
#         my_file = export_graphviz(tree_in_forest, out_file = dot_data)
#         graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#         graph.write_png('tree_' + str(i_tree) + '.png')
    # i_tree = i_tree + 1
