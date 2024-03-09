import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



df = pd.DataFrame(pd.read_csv("./allGameWeekDataMergedCleaned.csv"))

columns_for_calculation = ['name', 'goals_scored', 'total_points','creativity','assists','influence','clean_sheets','selected','yellow_cards']
selected_df = df[columns_for_calculation]
result = selected_df.groupby('name').mean().reset_index()
result.columns = ['name', 'averageofgoals', 'averageofpoints','AverageCreativity','AverageAssists','AverageThreat','AverageCS','AverageSel','AverageGC']

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

#graph 1
x1 = np.array(result['averageofgoals']).reshape(-1, 1)
y1 = np.array(result['averageofpoints'])

regr1 = LinearRegression()
regr1.fit(x1, y1)
y_pred1 = regr1.predict(x1)


grph1=axs[0, 0]
grph1.scatter(x1, y1, color="black")
grph1.plot(x1, y_pred1, color='red', label='Linear Regression')
grph1.set_title('Average Points Made vs Average Goals Scored')
grph1.set_xlabel('Average Goals Scored')
grph1.set_ylabel('Average Points Made')
grph1.text(0, -2, f"Gradient: {regr1.coef_[0]}", fontsize=10)
grph1.text(0, -2.5, f"Intercept: {regr1.intercept_}", fontsize=10)
grph1.text(0, -3, f"R-squared value: {regr1.score(x1, y1)}", fontsize=10)
plt.subplots_adjust(hspace=1)


#graph 2

x2 = np.array(result['AverageCreativity']).reshape(-1, 1)
y2 = np.array(result['AverageAssists'])

regr2 = LinearRegression()
regr2.fit(x2, y2)
y_pred2 = regr2.predict(x2)


grph2=axs[0, 1]
grph2.scatter(x2, y2, color="black")
grph2.plot(x2, y_pred2, color='red', label='Linear Regression')
grph2.set_title('Average Assist vs Average Creativity')
grph2.set_xlabel('Average Creativity')
grph2.set_ylabel('Average Assist')
grph1.text(1, -2, f"Gradient: {regr2.coef_[0]}", fontsize=10)
grph1.text(1, -2.5, f"Intercept: {regr2.intercept_}", fontsize=10)
grph1.text(1, -3, f"R-squared value: {regr2.score(x2, y2)}", fontsize=10)
plt.subplots_adjust(hspace=1)

#graph 3

x3 = np.array(result['AverageThreat']).reshape(-1, 1)
y3 = np.array(result['AverageCS'])

regr3 = LinearRegression()
regr3.fit(x3, y3)
y_pred3 = regr3.predict(x3)


grph3=axs[1, 0]
grph3.scatter(x3, y3, color="black")
grph3.plot(x3, y_pred3, color='red', label='Linear Regression')
grph3.set_title('Average Assist vs Average Creativity')
grph3.set_xlabel('Average Creativity')
grph3.set_ylabel('Average Assist')
grph3.text(0, -0.15, f"Gradient: {regr3.coef_[0]}", fontsize=10 )
grph3.text(0, -0.175, f"Intercept: {regr3.intercept_}", fontsize=10)
grph3.text(0, -0.2, f"R-squared value: {regr3.score(x3, y3)}", fontsize=10)
plt.subplots_adjust(hspace=1)

#graph 4

x4 = np.array(result['AverageSel']).reshape(-1, 1)
y4 = np.array(result['AverageGC'])

regr4 = LinearRegression()
regr4.fit(x4, y4)
y_pred4 = regr4.predict(x4)


grph4=axs[1, 1]
grph4.scatter(x4, y4, color="black")
grph4.plot(x4, y_pred4, color='red', label='Linear Regression')
grph4.set_title('Average Selecting vs Average Yellow Cards')
grph4.set_xlabel('Average Selecting')
grph4.set_ylabel('Average Yellow Cards')
grph3.text(40, -0.15, f"Gradient: {regr4.coef_[0]}", fontsize=10 )
grph3.text(40, -0.175, f"Intercept: {regr4.intercept_}", fontsize=10)
grph3.text(40, -0.2, f"R-squared value: {regr4.score(x4, y4)}", fontsize=10)
plt.subplots_adjust(hspace=1)


plt.show()



result.to_csv('regressionStuff.csv', index=False)