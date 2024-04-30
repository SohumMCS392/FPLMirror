import numpy as np
# Data Processing
import pandas as pd
import numpy as np
import matplotlib as plt
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image

#our own implementation of randomforrest
dataf = pd.DataFrame(pd.read_csv("./allGameWeekDataMergedCleaned.csv"))
class Node:
    #Each node will represent a node in the decision tree
    def __init__(self,gini, num_samples, num_samples_perclass, pred_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_perclass
        self.predicted_class = pred_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
class DecisionTree:
    def __init__(self, max_depth = None, min_samples_leaf = 1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
    def predict(self, X):
        return [self._predict(i) for i in X]
    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))
    def _best_split(self, X, y):
        (m, n) = X.shape
        if m <= self.min_samples_leaf:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return (best_idx, best_thr)
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            self._gini(y),
            len(y),
            num_samples_per_class,
            predicted_class,
        )
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

def gnb2023(data):
    #average data by player and keep stats we want to analyse (also add std of points scored)
    averages = data.groupby('name').agg({
        'total_points': ['mean', 'std'],  
        'goals_scored': 'mean', 
        'creativity': 'mean',
        'influence': 'mean' 
    })

    #averageOfAveragepoitns scored in league so far
    leagueaveragepoints = averages['total_points']['mean'].mean()

    #get the above aberage players 
    aboveAveragePlayers = averages[averages['total_points']['mean'] > leagueaveragepoints]


    #function that returns true of std is below 3.5 consistent
    def type_constant(std):
        if std < 3.5:
            return True
        else:
            return False

    #add a new column to average players indicating if they are a consistent player
    aboveAveragePlayers['typeConst'] = aboveAveragePlayers['total_points']['std'].apply(type_constant)

    #print(((aboveAveragePlayers['typeConst'] == True).sum())/len(aboveAveragePlayers))
    #print(((aboveAveragePlayers['typeConst'] == False).sum())/len(aboveAveragePlayers))
    #print(len(aboveAveragePlayers))

    #shuffle players
    aboveAveragePlayers_shuffled = aboveAveragePlayers.sample(frac = 1)

    #choose half for training data and the other hald is testing data
    half_len = len(aboveAveragePlayers_shuffled) // 2
    training = aboveAveragePlayers_shuffled.iloc[:half_len]
    test = aboveAveragePlayers_shuffled.iloc[half_len:]

    #print details 
    #print("training:")
    #print(training)
    #print(len(training[training['total_points']['std'] > training['total_points']['std'].mean()]))
    #print("TrainingCons", (training['typeConst'] == True).sum())
    #print("TrainingRisk", (training['typeConst'] == False).sum())


    #print("\ntesting:")
    #print(test)
    #print(len(test[test['total_points']['std'] > test['total_points']['std'].mean()]))
    #print("testingCons", (test['typeConst'] == True).sum())
    #print("testingRisk", (test['typeConst'] == False).sum())

    X_train = training[[('creativity', 'mean'), ('influence', 'mean'), ('goals_scored', 'mean')]]
    y_train = training['typeConst']
    X_test = test[[('creativity', 'mean'), ('influence', 'mean'), ('goals_scored', 'mean')]]
    y_test = test['typeConst']

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)


    true_positive = conf_matrix[1][1]
    false_positive = conf_matrix[0][1]
    false_negative = conf_matrix[1][0]
    true_negative = conf_matrix[0][0]

    #total_predictions = true_positive + false_positive + false_negative + true_negative

    totalConsPredicted = true_positive + false_positive
    totalConsPredictedCorrectly = true_positive
    totalRiskPredicted = false_negative + true_negative
    totalRiskPredictedCorrectly = true_negative

    true_predicted_true = (true_positive / totalConsPredicted) * 100
    true_predicted_false = (false_positive / totalConsPredicted) * 100
    false_predicted_true = (false_negative / totalRiskPredicted) * 100
    false_predicted_false = (true_negative / totalRiskPredicted) * 100

    """
    print("\n")

    print("totalConsPredicted:", totalConsPredicted)
    print("totalConsPredictedCorrectly:", totalConsPredictedCorrectly)

    print("")

    print("totalRiskPredicted:", totalRiskPredicted)
    print("totalRiskPredictedCorrectly:", totalRiskPredictedCorrectly)
    print("\n")


    print("Percentage of Cons predicted Correctly:", true_predicted_true)
    print("Percentage of Cons predicted Incorrectly:", true_predicted_false)
    print("Percentage of Risk predicted Correctly:", false_predicted_false)
    print("Percentage of Risk predicted Incorrectly:", false_predicted_true)

    """

    #first index is the number of con and Risk secodn is the correctness of naives
    return([true_predicted_true,false_predicted_false])



# CorCons = np.array([])
# CorRisk = np.array([])

# for i in range (0,1000):
#     CorCons = np.append(CorCons,gnb2023(dataf)[0])
#     CorRisk = np.append(CorRisk,gnb2023(dataf)[1])
#     if (i%100==0):
#         print(i/1000)

# print("percentage of Correct Catogrised Consistent Players", np.mean(CorCons))
# print("percentage of Correct Catogrised Risky Players",np.mean(CorRisk))


gnb2023(dataf)


# dataf = pd.DataFrame(pd.read_csv("./allHistory.csv"))

# def gnbAll(df):
#     #seasonEntries = [(0,306),(307,618),(619,925),(926,1234),(1235,1541),(1542,1852),(1853,2199),(2200,2538)]
    
#     NoSeason = np.array([])
#     PercentageCorrect = np.array([])
#     #for i in range(6,-1,-1):
#         #NoSeason=np.append(NoSeason,7-i)
#         #startrowindex = seasonEntries[i][0]
#         #endrowindex = 2199
#         #trainingDF = df.iloc[startrowindex:endrowindex+1]
#         #testingDF = df.iloc[2200 :2538+1]

#     trainingDF = df
#     for i in range(22,15,-1):
#         NoSeason=np.append(NoSeason,i)
#         trainingDF = trainingDF._append(df[df['season'] == f'20{i}-{i+1}'], ignore_index=True)
#         #trainingDF=df[df['season'] == f'20{i}-{i+1}']
#         testingDF=df[df['season'] == f'2023-24']
#         #averageOfAveragepoitns scored in league so far
#         leagueaveragepoints = trainingDF['mean'].mean()
    
#         #get the above aberage players 
#         trainingDF = trainingDF[trainingDF['mean'] > leagueaveragepoints]

#         #function that returns true of std is below 3.5 consistent
#         def type_constant(std):
#             if std < 3.5:
#                 return True
#             else:
#                 return False

#         #add a new column to average players indicating if they are a consistent player
#         trainingDF['typeConst'] = trainingDF['std'].apply(type_constant)
#         testingDF['typeConst'] = testingDF['std'].apply(type_constant)
#         #print(((aboveAveragePlayers['typeConst'] == True).sum())/len(aboveAveragePlayers))
#         #print(((aboveAveragePlayers['typeConst'] == False).sum())/len(aboveAveragePlayers))
#         #print(len(aboveAveragePlayers))

#         #print details 
#         #print("training:")
#         #print(training)
#         #print(len(trainingDF[trainingDF['std'] > trainingDF['std'].mean()]))
#         #print("TrainingCons", (training['typeConst'] == True).sum())
#         #print("TrainingRisk", (training['typeConst'] == False).sum())


#         #print("\ntesting:")
#         #print(test)
#         #print(len(testingDF[testingDF['std'] > testingDF['std'].mean()]))
#         #print("testingCons", (test['typeConst'] == True).sum())
#         #print("testingRisk", (test['typeConst'] == False).sum())
        
        
#         X_train = trainingDF[['creativity', 'influence', 'goals_scored']]
#         y_train = trainingDF['typeConst']
#         X_test = testingDF[['creativity', 'influence', 'goals_scored']]
#         y_test = testingDF['typeConst']
#         nb = NaiveBayes()
#         y_pred = (GaussianNB().fit(X_train, y_train)).predict(X_test)
#         (nb.fit(X_train, y_train))
#         y_pred2 = nb.predict(X_test)
#         conf_matrix = confusion_matrix(y_test, y_pred2)


#         true_positive = conf_matrix[1][1]
#         false_positive = conf_matrix[0][1]
#         false_negative = conf_matrix[1][0]
#         true_negative = conf_matrix[0][0]

#         #total_predictions = true_positive + false_positive + false_negative + true_negative



#         totalConsPredicted = true_positive + false_positive
#         totalConsPredictedCorrectly = true_positive
#         totalRiskPredicted = false_negative + true_negative
#         totalRiskPredictedCorrectly = true_negative

#         true_predicted_true = (true_positive / totalConsPredicted) * 100
#         true_predicted_false = (false_positive / totalConsPredicted) * 100
#         false_predicted_true = (false_negative / totalRiskPredicted) * 100
#         false_predicted_false = (true_negative / totalRiskPredicted) * 100
        

#         """
#         print("\n")

#         print("totalConsPredicted:", totalConsPredicted)
#         print("totalConsPredictedCorrectly:", totalConsPredictedCorrectly)

#         print("")

#         print("totalRiskPredicted:", totalRiskPredicted)
#         print("totalRiskPredictedCorrectly:", totalRiskPredictedCorrectly)
#         print("\n")


#         print("Percentage of Cons predicted Correctly:", true_predicted_true)
#         print("Percentage of Cons predicted Incorrectly:", true_predicted_false)
#         print("Percentage of Risk predicted Correctly:", false_predicted_false)
#         print("Percentage of Risk predicted Incorrectly:", false_predicted_true)

#         """
#         PercentageCorrect=np.append(PercentageCorrect,(true_predicted_true,false_predicted_false))
#         #first index is the number of con and Risk secodn is the correctness of naives
#     return(NoSeason,PercentageCorrect)

# x,y=gnbAll(dataf)

# y1=np.array([])
# y2=np.array([])
# for i in range(0,len(y)//2):
#     y1=np.append(y1,y[2*i])
#     y2=np.append(y2,y[2*i+1])

# #print(len(x))
# #print(len(y))
# for i in range (0,len(x)):
#     print("SeasonsUsed: ", x[i])
#     print("PercentageCorrectCon: ",y1[i])
#     print("PercentageCorrectRisk: ",y2[i])

# plt.plot(23-x, y1, marker='o', linestyle='-',label='Consistent Players')
# plt.plot(23-x, y2, marker='x', linestyle='-',label='Risky Players')
# plt.xlabel('Number of preceeding seasons used in training data')
# plt.ylabel('Precentage of correct categorisations')

# #plt.plot(x, y1, marker='o', linestyle='-')
# #plt.plot(x, y2, marker='x', linestyle='-')


# # Add labels and title
# plt.title('Percentage of correct predictions based on number of prior seasons as training data')

# # Show the plot
# plt.grid(True)
# plt.show()
# plt.legend()
    