import seaborn as sns;
df = sns.load_dataset('tips');
print(df.head());

y = df['tip']
X = df[df.columns.difference(['tip'])]
print(X.head())
print(df.info());

X['day'] = X['day'].cat.codes
X['sex'] = X['sex'].cat.codes
X['smoker'] = X['smoker'].cat.codes
X['time'] = X['time'].cat.codes

print(X)

from sklearn.model_selection import train_test_split;
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=0)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=200).fit(X_train,y_train)

from shapash.explainer.smart_explainer import SmartExplainer

xpl = SmartExplainer(model=reg)

xpl.compile(x=X_test)

print(xpl.to_pandas(max_contrib=3).head(10))

app = xpl.run_app(title_story="lol")



