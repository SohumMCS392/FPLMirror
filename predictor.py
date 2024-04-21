import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None 

#open data as csv
df = pd.read_csv('allGameWeekDataMergedCleaned.csv')
df_lasyYearAvg = pd.read_csv('lastSeasonAverage.csv')

#basicpicker

def firstPicker():
    #for the time being add another column to represent the total points picked
    df['total_points_predicted*'] = df['total_points']

    columns = ['Point Metric'] + [f"Week {i}" for i in range(0,34)]
    weeklypoints = pd.DataFrame(columns=columns)

    weeklypoints.loc['Top 11 Points'] = ['Top 11 Points'] + [0] * 34
    weeklypoints.loc['Median 11 points'] = ['Median 11 points'] + [0] * 34
    weeklypoints.loc['Average Random 11 points'] = ['Average Random 11 points'] + [0] * 34

    for week in range (1,34):
        weekData = df[df["game_week"] == week]
        weektoppoints=weekData.sort_values(by='total_points_predicted*', ascending=False).head(11)['total_points_predicted*'].sum()
 
        weekData['distanceFromMedian'] = abs(weekData['total_points_predicted*'] - weekData['total_points_predicted*'].median())
        weekMedianPoints = (weekData.sort_values(by='distanceFromMedian').head(11))['total_points_predicted*'].sum()

        average_points = np.array([])

        for i in range (0,50):
            randomSelection =  weekData['total_points_predicted*'].sample(11).sum()
            average_points = np.append(average_points, randomSelection)

        weekRandomPoints = np.mean(average_points)

        print(weeklypoints.head())
        print(week, weeklypoints.iat[0, week], weektoppoints, weeklypoints.iat[0, week] + weektoppoints  )
        weeklypoints.at['Top 11 Points', f"Week {week}"] = (weeklypoints.iat[0, week] + weektoppoints)
        weeklypoints.at['Median 11 points', f"Week {week}"] = (weeklypoints.iat[1, week] + weekMedianPoints)
        weeklypoints.at['Average Random 11 points', f"Week {week}"] = (weeklypoints.iat[2, week] + weekRandomPoints)

    print(weeklypoints)


firstPicker()

