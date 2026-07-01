import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
human_data = pd.read_table("human_data.txt");
print(human_data.head());
chimp_Data = pd.read_table("chimp_data.txt");
dog_data = pd.read_table("dog_data.txt");
from IPython.display import Image;

def kmers(sequence, size=6):
    return [sequence[x : x + size].lower() for x in range(len(sequence) - size + 1)];

human_data['words'] = human_data.apply(lambda x : kmers(x['sequence']),axis = 1);
human_data = human_data.drop('sequence', axis = 1);
chimp_Data['words'] = chimp_Data.apply(lambda x : kmers(x['sequence']),axis = 1);
chimp_Data = chimp_Data.drop('sequence', axis = 1);
dog_data['words'] = dog_data.apply(lambda x : kmers(x['sequence']),axis = 1);
dog_data = dog_data.drop('sequence', axis = 1);

print(human_data.head(3));

human_texts = list(human_data['words']);
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item]);
y_data = human_data.iloc[:,0].values;
chimp_text = list(chimp_Data['words']);
for item in range(len(chimp_text)):
    chimp_text[item] = ' '.join(chimp_text[item]);
y_data = human_data.iloc[:,0].values;
dog_text = list(dog_data['words']);
for item in range(len(dog_data)):
    dog_text[item] = ' '.join(dog_text[item]);
y_data = human_data.iloc[:,0].values;
from sklearn.feature_extraction.text import CountVectorizer;

cv = CountVectorizer(ngram_range=(4,4));
x = cv.fit_transform(human_texts);
x_chimp = cv.transform(chimp_text);
x_dog = cv.transform(dog_text)
print(human_texts[0]);
human_data['class'].value_counts().sort_index().plot.bar()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y_data, 
                                                    test_size = 0.20, 
                                                    random_state=42)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))



