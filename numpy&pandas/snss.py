import seaborn as sns;
import matplotlib.pyplot as plt;
# data from restaruant and features like data,time
# we can see that tip is dependent feature 
# while sex,time is not depend i.e, independent
# we are going to use pairdist
df = sns.load_dataset("tips");
# # correaltion is use to see the relation between the two variables i,e, features
# # values lies between -1 and 1 it is pearson coefficient
# print(df.head(2));
# # print(df.corr())
# print(df.info())
# print(df.dtypes)
# print(df.corr(numeric_only=True))

# print(sns.heatmap(df.corr(numeric_only=True)))
# # plt.show();

# # ======== JoinPlot ==========
# #univariate analysis

# print(sns.jointplot(x="tip",y="total_bill",data=df,kind='reg'))
# # plt.show();

# #pair plot it is scatter plot in which one variable in the same data is matched with another variable value like this pair plots are just elaborations on this

# sns.pairplot(df)
# # plt.show();
# sns.pairplot(df,hue="sex")
# # plt.show();

# # dist plot is used to create histograms we can see the data distribution

# sns.displot(df['tip']);
# # plt.show();

#===================== Categorical Plota ===========================
# Boxplt
# violinplot
# countplot

# sns.countplot(x='sex',data=df)
# plt.show();
# sns.barplot(x="total_bill",y='sex',data=df)
# plt.show();
# sns.boxplot(x='smoker',y='total_bill',data=df)
# plt.show();

# sns.violinplot(x='total_bill',y='day',data=df,palette='rainbow')
# plt.show();

df = sns.load_dataset('iris');
print(df.info());

sns.countplot(x='sepal_length',hue='species',data=df)
plt.show();

sns.jointplot(x='sepal_length',y='sepal_width',hue='species',data=df);
plt.show();
