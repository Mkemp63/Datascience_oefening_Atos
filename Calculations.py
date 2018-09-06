import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split


class Calculate(object):

    def __init__(self, ddofnumber, csvstring, columnname='price', correlationname='overall_satisfaction'):
        self.ddofnumber = ddofnumber
        self.csvstring = csvstring
        self.columnname = columnname
        self.correlationname = correlationname

    def createdf(self):
        global df
        global group1
        global group2
        df = pd.read_csv(self.csvstring)
        group1, group2 = train_test_split(df, test_size=0.5)

    def mean(self):
        if self.columnname != '':
            n = len(df)
            if n < 1:
                raise ValueError('mean requires at least one data point')
            print("the mean of ", self.columnname, " is: ", sum(df[self.columnname]) / n)
            return sum(df[self.columnname]) / n
        else:
            print("specify columnname in objectcall for standard deviation")

    def squaredev(self):
        c = self.mean()
        sd = sum((x - c) ** 2 for x in df[self.columnname])
        print("the squared deviation from the mean is", sd)
        return sd

    def stddev(self):
        n = len(df[self.columnname])
        if n < 2:
            raise ValueError('variance requires at least two data points')
        ss = self.squaredev()
        pvar = ss / (n - self.ddofnumber)
        print("the standard deviation is: ", pvar ** 0.5)
        return pvar ** 0.5

    def ttest(self):
        n = len(group1)
        t = (group1[self.columnname].mean() - group2[self.columnname].mean()) / (self.stddev() * np.sqrt(2 / n))
        degree = 2 * n - 2
        p = 1 - stats.t.cdf(t, df=degree)
        # values will not be the same if script is run multiple times due to sample "error"
        print("")
        print("T-test values:")
        print("t = " + str(t))
        print("p = " + str(2 * p))
        return t

    def median(self):
        n = len(df[self.columnname])
        if n % 2 == 0:
            i = n / 2
            median = df.iloc[int(i)]['price']
            print(median)
        elif n % 2 != 0:
            i = n / 2
            a = i + 0.5
            b = i - 0.5
            total = df.iloc[int(a)]['price'] + df.iloc[int(b)]['price']
            median = total / 2
        print("the median of ", self.columnname, " is ", median)
        return median

    def correlation(self):
        print("")
        print("the correlationmatrix between x =", self.columnname, " and y =", self.correlationname, " is:")
        print(np.corrcoef(df[self.columnname], df[self.correlationname]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def scatterplot(self):
        sp = sns.scatterplot(x="longitude", y="latitude", hue="room_type", data=df)
        plt.grid()
        sp.set_title("scatterplot latitude longitude")
        plt.ylim(52.275, 52.45)
        plt.xlim(4.75, 5.05)
        plt.show()

    def boxplot(self):
        df["per person per night"] = df["price"] / df["accommodates"]
        sns.catplot(x="per person per night", kind="box", row="room_type", orient="h", data=df)
        plt.grid()
        # Left out datapoints beyond q4 on purpose. Purpose of graph is to show the box, not the values beyond q4
        plt.xlim(0, 125)
        plt.show()

    def barplot(self):
        bp = sns.catplot(x="room_type", y="overall_satisfaction", kind="bar", data=df)
        bp.set_titles("Barplot of roomtype against price")
        plt.show()

    def countplot(self):
        cp = sns.catplot(x="room_type", kind="count", order=df['room_type'].value_counts().index, data=df)
        cp.set_titles("Countplot of roomtypes")
        plt.show()

    def sigmoidplot(self):
        # i did not succeed in making a sigmoid plot from dataset data, so we will use a numpy array to demonstrate
        # functionality
        x = np.arange(-10, 10, 0.2)
        plt.plot(x, self.sigmoid(x))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()

    def runall(self):
        self.createdf()
        self.median()
        self.ttest()
        self.correlation()
        self.scatterplot()
        self.barplot()
        self.boxplot()
        self.countplot()
        self.sigmoidplot()
