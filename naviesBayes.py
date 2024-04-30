import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import mean_squared_error, r2_score
pd.options.mode.chained_assignment = None

#prior cleaned data
dataf = pd.DataFrame(pd.read_csv("./allGameWeekDataMergedCleaned.csv"))


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

    y_pred = (GaussianNB().fit(X_train, y_train)).predict(X_test)

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



CorCons = np.array([])
CorRisk = np.array([])

for i in range (0,1000):
    CorCons = np.append(CorCons,gnb2023(dataf)[0])
    CorRisk = np.append(CorRisk,gnb2023(dataf)[1])
    if (i%100==0):
        print(i/1000)

print("percentage of Correct Catogrised Consistent Players", np.mean(CorCons))
print("percentage of Correct Catogrised Risky Players",np.mean(CorRisk))


gnb2023(dataf)


dataf = pd.DataFrame(pd.read_csv("./allHistory.csv"))

def gnbAll(df):


    #seasonEntries = [(0,306),(307,618),(619,925),(926,1234),(1235,1541),(1542,1852),(1853,2199),(2200,2538)]
    
    NoSeason = np.array([])
    PercentageCorrect = np.array([])
    #for i in range(6,-1,-1):
        #NoSeason=np.append(NoSeason,7-i)
        #startrowindex = seasonEntries[i][0]
        #endrowindex = 2199
        #trainingDF = df.iloc[startrowindex:endrowindex+1]
        #testingDF = df.iloc[2200 :2538+1]

    trainingDF = df
    for i in range(22,21,-1):
        NoSeason=np.append(NoSeason,i)
        trainingDF = trainingDF._append(df[df['season'] == f'20{i}-{i+1}'], ignore_index=True)
        #trainingDF=df[df['season'] == f'20{i}-{i+1}']
        testingDF=df[df['season'] == f'2023-24']
        #averageOfAveragepoitns scored in league so far
        leagueaveragepoints = trainingDF['mean'].mean()
    
        #get the above aberage players 
        trainingDF = trainingDF[trainingDF['mean'] > leagueaveragepoints]

        #function that returns true of std is below 3.5 consistent
        def type_constant(std):
            if std < 3.5:
                return True
            else:
                return False

        #add a new column to average players indicating if they are a consistent player
        trainingDF['typeConst'] = trainingDF['std'].apply(type_constant)
        testingDF['typeConst'] = testingDF['std'].apply(type_constant)
        #print(((aboveAveragePlayers['typeConst'] == True).sum())/len(aboveAveragePlayers))
        #print(((aboveAveragePlayers['typeConst'] == False).sum())/len(aboveAveragePlayers))
        #print(len(aboveAveragePlayers))

        #print details 
        #print("training:")
        #print(training)
        #print(len(trainingDF[trainingDF['std'] > trainingDF['std'].mean()]))
        #print("TrainingCons", (training['typeConst'] == True).sum())
        #print("TrainingRisk", (training['typeConst'] == False).sum())


        #print("\ntesting:")
        #print(test)
        #print(len(testingDF[testingDF['std'] > testingDF['std'].mean()]))
        #print("testingCons", (test['typeConst'] == True).sum())
        #print("testingRisk", (test['typeConst'] == False).sum())
        
        
        X_train = trainingDF[['creativity', 'influence', 'goals_scored']]
        y_train = trainingDF['typeConst']
        X_test = testingDF[['creativity', 'influence', 'goals_scored']]
        y_test = testingDF['typeConst']

        y_pred = (GaussianNB().fit(X_train, y_train)).predict(X_test)

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
        PercentageCorrect=np.append(PercentageCorrect,(true_predicted_true,false_predicted_false))

        #first index is the number of con and Risk secodn is the correctness of naives
    return(NoSeason,PercentageCorrect)

x,y=gnbAll(dataf)

y1=np.array([])
y2=np.array([])
for i in range(0,len(y)//2):
    y1=np.append(y1,y[2*i])
    y2=np.append(y2,y[2*i+1])

#print(len(x))
#print(len(y))
for i in range (0,len(x)):
    print("SeasonsUsed: ", x[i])
    print("PercentageCorrectCon: ",y1[i])
    print("PercentageCorrectRisk: ",y2[i])

plt.plot(23-x, y1, marker='o', linestyle='-',label='Consistent Players')
plt.plot(23-x, y2, marker='x', linestyle='-',label='Risky Players')
plt.xlabel('Number of preceeding seasons used in training data')
plt.ylabel('Precentage of correct categorisations')

#plt.plot(x, y1, marker='o', linestyle='-')
#plt.plot(x, y2, marker='x', linestyle='-')


# Add labels and title
plt.title('Percentage of correct predictions based on number of prior seasons as training data')

# Show the plot
plt.grid(True)
plt.show()
plt.legend()




"""
Target 95% confident with error of 5% std off mean


WRONG FROMULA
n = z^2 * p * (1-p) * E^2
z = 1.645 z^2 = 2.706025 p = 0.551 (1-p) = 0.448 E = 0.05 E^2 = 0.0025


CORRECT
n= Z^2 std^2 / E^2 

alpha (significance level) = 0.05
(1-alpha) = 0.95 = confidence interval

Z=1.96
Z ^2 =1.96^2 =3.8416


std (con) = 50
std (ris) = 100
"""

#print(averages)
#print(aboveAveragePlayers_shuffled)
#averages.to_csv('naives.csv', index=False)


