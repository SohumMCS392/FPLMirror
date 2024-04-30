import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer




df = pd.read_csv("allGameWeekDataMergedCleaned.csv")
lastYearAv = pd.read_csv("lastSeasonAverage.csv")

def initalPca(df):

    featuresUsed =["xP","assists","bonus","bps","clean_sheets","creativity","element","expected_assists","expected_goal_involvements","expected_goals","expected_goals_conceded","goals_conceded","goals_scored","minutes","own_goals","penalties_missed","penalties_saved","red_cards","saves","selected","threat","total_points","value"]

    #standardise data:
    standardisedFeats = StandardScaler().fit_transform(df[featuresUsed])

    #intial view
    fullPCA = PCA().fit(standardisedFeats)

    varienceRatioSum = np.cumsum(fullPCA.explained_variance_ratio_)

    requiredComponents = np.where(varienceRatioSum >= 0.8)[0][0] + 1 

    print(requiredComponents)

    plt.figure(figsize=(10, 6))
    plt.plot(varienceRatioSum)
    plt.xlabel('Number of Components')
    plt.ylabel('Varience Ratio Sum')
    plt.axhline(y=0.80, color='r', linestyle='--')
    plt.axvline(x=requiredComponents, color='r', linestyle='--')
    plt.show()

#initalPca()

def topComponentsPca(df):

    #df.set_index(['name', 'position', 'team', 'game_week'], inplace=True)

    featuresUsed = ["xP","assists","bonus","bps","clean_sheets","creativity","element","expected_assists","expected_goal_involvements","expected_goals","expected_goals_conceded","goals_conceded","goals_scored","minutes","own_goals","penalties_missed","penalties_saved","red_cards","saves","selected","threat","total_points","value"]

    sclr = StandardScaler()
    #standardise data:
    standardisedFeats = sclr.fit_transform(df[featuresUsed])

    #80%
    pca = PCA(n_components=10)
    pca.fit(standardisedFeats)

    pca80Mat = pca.components_

    pcaNewData = pca.fit_transform(standardisedFeats)

    pca80df = pd.DataFrame(pcaNewData, columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10'])
    
    dftofit = pd.concat([df[['name', 'position','team', 'total_points']].reset_index(drop=True), pca80df], axis=1)

    #print(dftofit)

    return(pca80Mat, dftofit,sclr)

#initalPca()

"""
#DONOT EDIT -> TOOK LONG TIME TO GET WORKING
def nextWeekForecastArima(df, week):
    statsToUse = ["xP", "assists", "bonus", "bps", "clean_sheets", "creativity", 
                  "element", "expected_assists", "expected_goal_involvements", 
                  "expected_goals", "expected_goals_conceded", "goals_conceded", 
                  "goals_scored", "minutes", "own_goals", "penalties_missed", 
                  "penalties_saved", "red_cards", "saves", "selected", "threat", 
                  "total_points", "value"]

    predictions = []
    adfResults = []

    print("week " + str(week))


    for player in df['name'].unique():
        playerStats = []
        playerAdfs = []

        for stat in statsToUse:
            series = df[df['name'] == player][stat].astype(float).dropna()

            print("player "+ str(player))


            if len(series) >= 5:
                seriesDiff = series.diff().dropna()

                print(seriesDiff)

                if len(series) >= 10:
                    result = adfuller(seriesDiff)
                    playerAdfs.append((stat, result[0], result[1]))
                    print(result)

                model = auto_arima(series, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
                forecast = model.predict(n_periods=1)
                print("here")
                print(type(forecast))
                print("here-")

                playerStats.append(forecast.iloc[0])  

            else:
                playerStats.append(np.nan) 

        predictions.append([player] + playerStats)
        adfResults.append((player, playerAdfs))

    columns = ['name'] + statsToUse
    return pd.DataFrame(predictions, columns=columns), dict(adfResults)



def linRegressionArima(df):

    transformMat, dataset = initalPca(df)

    xMat = dataset[[f'pc{i}' for i in range(1, 11)]]
    yVec = dataset['total_points']
    
    model = LinearRegression()
    model.fit(xMat, yVec)

    weights = model.coef_ 





def 
    for week in range (1,34):
        if week == 1:
            print("todo")
        else:
            priorweeks = df[df['game_week'] < week]
            priorweeksAv = priorweeks.groupby('name').agg({'name': 'first','position': 'first','team' : 'first','xP' : 'mean','assists': 'mean','bonus': 'mean','bps': 'mean','clean_sheets': 'mean','creativity': 'mean','element': 'mean','expected_assists': 'mean','expected_goal_involvements': 'mean','expected_goals': 'mean','expected_goals_conceded': 'mean','goals_conceded': 'mean','goals_scored': 'mean','minutes': 'mean','own_goals': 'mean','penalties_saved': 'mean','red_cards': 'mean','saves': 'mean','penalties_missed': 'mean','selected': 'mean','threat': 'mean','total_points': 'mean','value': 'mean'})
            
            forecast, adfs = nextWeekForecast(priorweeks, week)

            thisWeekPredict = pd.DataFrame(priorweeks['name'].unique(), columns=['name'])

            thisWeekPredict = thisWeekPredict.merge(forecast, on='name', how='left')

            #print(adfs)
            
            if week == 7:
                print(thisWeekPredict)
                print(thisWeekPredict["threat"])
                assert(False)
"""

"""  
def nextWeekForecast(df,week):

    statsToUse = ["xP", "assists", "bonus", "clean_sheets", "creativity",  "goals_conceded", 
                  "goals_scored", "minutes", "own_goals", "penalties_missed", 
                  "penalties_saved", "red_cards", "saves", "value"]

    predictions = pd.DataFrame()
    predictions["names"]=df["name"].unique()


    for names in predictions['names']:

        df = df.sort_values(by='game_week')

        for stat in statsToUse:

            playStat = df[df['name'] == names][stat].astype(float).dropna().reset_index(drop=True)
            
            laggedData = pd.DataFrame(playStat) 
            
            for i in range(0, week):
                laggedData["lag_" + str(i)] = playStat.shift(i)

            #avVal = lastYearAv[lastYearAv['name'] == names][stat]
            #laggedData.fillna(avVal, inplace=True)
            laggedData.ffill(inplace=True)

            print(laggedData)

    
def forXgAndForest(df):
    for week in range (1,5):
        print(week)
        priorweeks = df[df['game_week'] <= week]
        nextWeekForecast(priorweeks,week)  

"""

def ewma(df,week,statsToUse):

    predictions = pd.DataFrame()
    predictions["names"]=df["name"].unique()

    players = predictions["names"].unique().tolist()

    for stat in statsToUse:

        statPredictions = []
        
        for player in players:
            
            player_data = df[(df['name'] == player) & (df['game_week'] <= week)]

            if not player_data.empty:

                ewmaVal = player_data[stat].ewm(span=3, adjust=False).mean().iloc[-1]

            else:
                
                ewmaVal = None
            
            statPredictions.append(ewmaVal)
        
        predictions[stat] = statPredictions

    return predictions
    
def ewmaWeekly(df):

    weekPredictions = []

    statsToUse = ["xP","assists","bonus","bps","clean_sheets","creativity","element","expected_assists","expected_goal_involvements","expected_goals","expected_goals_conceded","goals_conceded","goals_scored","minutes","own_goals","penalties_missed","penalties_saved","red_cards","saves","selected","threat","total_points","value"]
    
    for week in range (1,33):
        print("Calculating EWMA for week" +str(week))
        predictions = ewma(df, week, statsToUse)
        #print(week)
        #print(predictions)
        #print(str(predictions.drop(columns=['names']).isna().all(axis=1).sum()))
        #print(predictions.iloc[200])

        weekPredictions.append(predictions)
    
    return (weekPredictions)

def linRegression(df):

    predictions = pd.DataFrame()
    predictions["names"]=df["name"].unique()
    weeklyEwma = ewmaWeekly(df)

    for week in range(1,34):

        print(week)

        if week == 1:
            predictions = predictions.merge(lastYearAv[['name', 'total_points']], how='left', left_on='names', right_on='name')
            predictions.rename(columns={'total_points': 'gw1 points'}, inplace=True)
            predictions.drop(columns='name', inplace=True)

        else:

            dataSoFar = df[df['game_week'] <= week]
            statsToUse = ["xP","assists","bonus","bps","clean_sheets","creativity","element","expected_assists","expected_goal_involvements","expected_goals","expected_goals_conceded","goals_conceded","goals_scored","minutes","own_goals","penalties_missed","penalties_saved","red_cards","saves","selected","threat","total_points","value"]
            dataSoFarAvg = dataSoFar.groupby(['name', 'position', 'team'])[statsToUse].mean().reset_index().sort_values(by='name')

            transformMat, dataset, sclr = topComponentsPca(dataSoFar)
            xMat = dataset[[f'pc{i}' for i in range(1, 11)]]
            yVec = dataset['total_points']
            
            model = LinearRegression()
            model.fit(xMat, yVec)

            weights = model.coef_ 

            thisWeekPredictions = weeklyEwma[week-2]

            thisWeekPredictions=thisWeekPredictions.sort_values(by='names')

            featuresUsed = ["xP","assists","bonus","bps","clean_sheets","creativity","element","expected_assists","expected_goal_involvements","expected_goals","expected_goals_conceded","goals_conceded","goals_scored","minutes","own_goals","penalties_missed","penalties_saved","red_cards","saves","selected","threat","total_points","value"]

            thisWeekPredictions = sclr.transform(thisWeekPredictions[featuresUsed])

            imputer = SimpleImputer(strategy='mean')

            thisWeekPredictions = imputer.fit_transform(thisWeekPredictions)

            thisWeekPredictions = pd.DataFrame(thisWeekPredictions, columns=featuresUsed)

            thisWeekPredictionsPca = np.dot(thisWeekPredictions, transformMat.T)

            predictedPoints = model.predict(thisWeekPredictionsPca)

            predictions["gw" + str(week) + " points"] = predictedPoints

    return(predictions)

x=linRegression(df)
x.to_csv('predictedPoints.csv', index=False) 
