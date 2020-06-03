import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

data = pd.DataFrame({'X1': [1,2,3,4,5], 
                     'X2': [2,3,4,5,6],
                     'X3': [2,6,3,7,1],
                     'X4': [3,1,3,5,2],
                     'Y' : [0,2,2,3,4]   })

print(data.head())

#Method 1 to Detect MultiCollinearity
plt.rcParams["figure.figsize"] = (8,4)
sns.heatmap(data.corr())
plt.show()

#Here we can see that X1 and X2 have a high and similar correlation coefficient
#(Also X3 and X4 have similar coefficients but they are lower so we can allow low collinearity)


#Method 2 to Detect MultiCollinearity

def get_VIF(dataFrame , target):
    X = add_constant(dataFrame.loc[:, dataFrame.columns != target])
    seriesObject = pd.Series([variance_inflation_factor(X.values,i) for i in range(X.shape[1])] , index=X.columns,)
    return seriesObject

target = 'Y'
print(get_VIF(data,target))

#Here we Observe that X1 and X2 are having VIF value of infinity so we need to drop one of them
#(Any value greater than 5-6 shows MultiCollinearity)

#Therefore we can drop X1 or X2