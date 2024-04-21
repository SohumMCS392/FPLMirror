import pandas as pd

def seasons2023():

    #intialise dataframe
    cummulativeDF = pd.DataFrame()

    #gets i week data and adds it to the cummulativeDF
    def add_data (cummulativeDF, currentWeekId):
        thisWeekDF = pd.DataFrame(pd.read_csv(f"./data/2023-24/gws/gw{currentWeekId}.csv"))
        thisWeekDF["game_week"] = currentWeekId 
        
        cummulativeDF = cummulativeDF._append(thisWeekDF, ignore_index=True)
        return cummulativeDF
    
    #Go through each week and add the data to the DF
    for i in range (1,38):
        try:
            cummulativeDF = add_data(cummulativeDF, i)
        except Exception as e:
            print(e)

    #find all player who have played more than half avalible games
    numAppearances= pd.merge(cummulativeDF.groupby('name').agg({'starts': 'sum'}).reset_index(), (cummulativeDF['name'].value_counts().reset_index()), on='name', how='outer')
    numAppearances['ratio'] = numAppearances['starts'] / numAppearances['count']
    numAppearances = numAppearances[numAppearances['ratio'] >= 1/2]

    #remove people who played less than half
    cummulativeDF = cummulativeDF[cummulativeDF['name'].isin(numAppearances['name'])]

    #output/save DF
    cummulativeDF.to_csv('allGameWeekDataMergedCleaned.csv', index=False)

def allSeasons():

    #intialise dataframe
    cummulativeDF = pd.DataFrame() 

    dfs_to_concat = []
    #gets i week data and adds it to the cummulativeDF
    def add_data (cummulativeDF, season, week):
        if season<19:
            encodingType = "cp1252"
        else:
            encodingType = "utf-8"

        thisWeekDF = pd.DataFrame(pd.read_csv(f"./data/20{season}-{season+1}/gws/gw{week}.csv",encoding = encodingType))
        thisWeekDF["season"] = (f"20{season}-{season+1}")
        thisWeekDF["game_week"] = week
        if 'minutes' in thisWeekDF.columns:
            if 'starts' not in thisWeekDF.columns:
                thisWeekDF['starts'] = (thisWeekDF['minutes'] > 0).astype(int)
            else:
                thisWeekDF['starts'] = thisWeekDF['starts'].fillna((thisWeekDF['minutes'] > 0).astype(int))

        return thisWeekDF

    for i in range (16,24):
        for j in range (1,39):
            print(str(i) + ":"+ str(j))
            try: 
                dfs_to_concat.append(add_data(cummulativeDF, i, j))
            except Exception as e:
                print(e)

    cummulativeDF = pd.concat(dfs_to_concat, ignore_index=True)

    #print(cummulativeDF)

    cummulativeDF.rename(columns={'total_points': 'mean'}, inplace=True)
    cummulativeDF["std"]=cummulativeDF["mean"]

    agg_funcs = {'starts': 'sum', 'mean': 'mean', 'std': 'std', 'creativity': 'mean', 'influence': 'mean', 'goals_scored':'mean'}

    
    #find all player who have played more than half avalible games
    numAppearances = pd.merge(
        cummulativeDF.groupby(['name', 'season']).agg(agg_funcs),
        cummulativeDF.groupby(['name', 'season']).size().reset_index(name='count'),
        on=['name', 'season'],
        how='outer'
    ) 



    numAppearances['ratio'] = numAppearances['starts'] / numAppearances['count']
    #print(numAppearances) 
    numAppearances = numAppearances[numAppearances['ratio'] >= 1/2]

    
    #remove people who played less than half
    numAppearancesFiltered = numAppearances[numAppearances['ratio'] >= 1/2]

    #output/save DF
    numAppearances.to_csv('allHistory.csv', index=False)

def average2022():

    #intialise dataframe
    cummulativeDF = pd.DataFrame()

    #gets i week data and adds it to the cummulativeDF
    def add_data (cummulativeDF, currentWeekId):
        thisWeekDF = pd.DataFrame(pd.read_csv(f"./data/2022-23/gws/gw{currentWeekId}.csv"))
        thisWeekDF["game_week"] = currentWeekId 
        
        cummulativeDF = cummulativeDF._append(thisWeekDF, ignore_index=True)
        return cummulativeDF
    
    #Go through each week and add the data to the DF
    for i in range (1,38):
        try:
            cummulativeDF = add_data(cummulativeDF, i)
        except Exception as e:
            print(e)


    #find all player who have played more than half avalible games
    numAppearances= pd.merge(cummulativeDF.groupby('name').agg({'starts': 'sum'}).reset_index(), (cummulativeDF['name'].value_counts().reset_index()), on='name', how='outer')
    numAppearances['ratio'] = numAppearances['starts'] / numAppearances['count']
    numAppearances = numAppearances[numAppearances['ratio'] >= 1/3]

    #remove people who played less than third
    cummulativeDF = cummulativeDF[cummulativeDF['name'].isin(numAppearances['name'])]

    #get averages
    cummulativeDF = cummulativeDF.groupby('name').agg({
    'name': 'first',
    'position': 'first',
    'team' : 'first',
    'xP' : 'mean',
    'assists': 'mean',
    'bonus': 'mean',
    'bps': 'mean',
    'clean_sheets': 'mean',
    'creativity': 'mean',
    'element': 'mean',
    'expected_assists': 'mean',
    'expected_goal_involvements': 'mean',
    'expected_goals': 'mean',
    'expected_goals_conceded': 'mean',
    'goals_conceded': 'mean',
    'goals_scored': 'mean',
    'minutes': 'mean',
    'opponent_team': 'mean',
    'own_goals': 'mean',
    'penalties_missed': 'mean',
    'red_cards': 'mean',
    'saves': 'mean',
    'penalties_missed': 'mean',
    'selected': 'mean',
    'threat': 'mean',
    'total_points': 'mean',
    'value': 'mean'
})

    #output/save DF
    cummulativeDF.to_csv('lastSeasonAverage.csv', index=False)
    

#allSeasons()
#seasons2023()
average2022()