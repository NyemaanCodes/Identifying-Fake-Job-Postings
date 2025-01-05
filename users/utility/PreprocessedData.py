import pandas as pd
from sklearn.model_selection import train_test_split
from django.conf import settings
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation

path = settings.MEDIA_ROOT + "//" + 'DataSet.csv'
data=pd.read_csv(path)
data.drop(['salary_range', 'telecommuting', 'has_company_logo', 'has_questions'],axis=1,inplace = True)
print(data.head())
data.fillna('', inplace=True)

# Create independent and Dependent Features
columns = data.columns.tolist()
# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["fraudulent"]]
# Store the variable we are predicting
target = "fraudulent"
# Define a random state
state = np.random.RandomState(42)
X = data[columns]
Y = data["fraudulent"]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)
from imblearn.under_sampling import RandomUnderSampler

under_sampler = RandomUnderSampler()
X_res, y_res = under_sampler.fit_resample(X, Y)

df1 = pd.DataFrame(X_res)

df3 = pd.DataFrame(y_res)

# the default behaviour is join='outer'
# inner join

result = pd.concat([df1, df3], axis=1, join='inner')
print(result)
data = result;

# Exploratary Data Analysis
labels = 'Fake', 'Real'
sizes = [data.fraudulent[data['fraudulent']== 1].count(), data.fraudulent[data['fraudulent']== 0].count()]
# explode = (0, 0.1)
# fig1, ax1 = plt.subplots(figsize=(8, 6)) #size of the pie chart
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
#         shadow=True, startangle=120) #autopct %1.2f%% for 2 digit precision
# ax1.axis('equal')
# plt.title("Proportion of Fraudulent", size = 7)
# plt.show()
def split(location):
    l = location.split(',')
    return l[0]

data['country'] = data.location.apply(split)
data['country']
# creating a dictionary(key-value pair) with top 10 country
country = dict(data.country.value_counts()[:11])
del country[''] #deleting country with space values
plt.figure(figsize=(12,9))
plt.title('Country-wise Job Posting', size=20)
plt.bar(country.keys(), country.values()) #(xaxis,yaxis)
plt.ylabel('No. of jobs', size=10)
plt.xlabel('Countries', size=10)
# plt.show()
# visualizing jobs based on experience
experience = dict(data.required_experience.value_counts())
del experience['']
plt.figure(figsize=(12,9))
plt.bar(experience.keys(), experience.values())
plt.title('No. of Jobs with Experience')
plt.xlabel('Experience', size=10)
plt.ylabel('No. of jobs', size=10)
plt.xticks(rotation=35)
# plt.show()
#Most frequent jobs
print(data.title.value_counts()[:10])
#Titles and count of fraudulent jobs
# checking for most fake jobs based on title
print(data[data.fraudulent==1].title.value_counts()[:10])
# For textual type data we will try to create word cloud
# but before that we will try to create text combining all the data present in
# our database.
data['text'] = data['title']+' '+data['location']+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']+' '+data['industry']

del data['title']
del data['location']
del data['department']
del data['company_profile']
del data['description']
del data['requirements']
del data['benefits']
del data['required_experience']
del data['required_education']
del data['industry']
del data['function']
del data['country']
del data['employment_type']

def preProcessed_data_view():
    return data.to_html
# we will plot 3 kind of word cloud
# 1st we will visualize all the words our data using the wordcloud plot
# 2nd we will visualize common words in real job posting
# 3rd we will visualize common words in fraud job posting
# join function is a core python function
from wordcloud import WordCloud
all_words = ''.join([text for text in data["text"]])
wordcloud = WordCloud(width = 800, height = 500, random_state=21, max_font_size=120).generate(all_words)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.show()
 # Common words in real job posting texts

# real_post = ''.join([text for text in data["text"][data['fraudulent']==0]])
# wordcloud = WordCloud(width = 800, height = 500, random_state=21, max_font_size=120).generate(real_post)
#
# plt.figure(figsize=(10,8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Common words in fraud job posting texts

fraud_post = ''.join([text for text in data["text"][data['fraudulent'] == 1]])

# wordcloud = WordCloud(width = 800, height = 500, random_state=21, max_font_size=120).generate(fraud_post)
# plt.figure(figsize=(10,8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# NLTK :: Natural Language Toolkit
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
#loading the stopwords
stop_words = set(stopwords.words("english"))
#converting all the text to lower case
data['text'] = data['text'].apply(lambda x:x.lower())
#removing the stop words from the corpus
data['text'] = data['text'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop_words)]))

from sklearn.model_selection import train_test_split
# Splitting dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.3)

# The model cannot operate text data so we need to convert our data into vector format
# we will be using Bag of words model
from sklearn.feature_extraction.text import CountVectorizer

#  instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix
# fit
vect.fit(X_train)

# transform training data
X_train_dtm = vect.transform(X_train)
# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def build_naive_bayes():
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_nb = nb.predict(X_test_dtm)
    accuracy_score(y_test, y_pred_nb)
    print("Classification Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Classification Report\n")
    nb_cr = classification_report(y_test, y_pred_nb,output_dict=True)
    print(classification_report(y_test, y_pred_nb))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_nb))
    cm = confusion_matrix(y_test, y_pred_nb)
    nb_cr = pd.DataFrame(nb_cr).transpose()
    nb_cr = pd.DataFrame(nb_cr)
    return nb_cr.to_html


def build_knn():
    knn = KNeighborsClassifier()
    knn.fit(X_train_dtm, y_train)
    y_pred_knn = knn.predict(X_test_dtm)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    print("Classification Accuracy:", knn_acc)
    print("Classification Report\n")
    knn_cr = classification_report(y_test, y_pred_knn,output_dict=True)
    print(classification_report(y_test, y_pred_knn))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_knn))
    cm = confusion_matrix(y_test, y_pred_knn)
    knn_cr = pd.DataFrame(knn_cr).transpose()
    knn_cr = pd.DataFrame(knn_cr)
    return knn_cr.to_html

def build_decsionTree():
    dt = DecisionTreeClassifier()
    dt.fit(X_train_dtm, y_train)
    y_pred_dt = dt.predict(X_test_dtm)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    print("Classification Accuracy:", dt_acc)
    print("Classification Report\n")
    dt_cr = classification_report(y_test, y_pred_dt,output_dict=True)
    print(classification_report(y_test, y_pred_dt))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_dt))
    cm = confusion_matrix(y_test, y_pred_dt)
    dt_cr = pd.DataFrame(dt_cr).transpose()
    dt_cr = pd.DataFrame(dt_cr)
    return dt_cr.to_html

def build_randomForest():
    dt = RandomForestClassifier()
    dt.fit(X_train_dtm, y_train)
    y_pred_dt = dt.predict(X_test_dtm)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    print("Classification Accuracy:", dt_acc)
    print("Classification Report\n")
    dt_cr = classification_report(y_test, y_pred_dt,output_dict=True)
    print(classification_report(y_test, y_pred_dt))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_dt))
    cm = confusion_matrix(y_test, y_pred_dt)
    dt_cr = pd.DataFrame(dt_cr).transpose()
    dt_cr = pd.DataFrame(dt_cr)
    return dt_cr.to_html

def build_svm():
    dt = SVC()
    dt.fit(X_train_dtm, y_train)
    y_pred_dt = dt.predict(X_test_dtm)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    print("Classification Accuracy:", dt_acc)
    print("Classification Report\n")
    dt_cr = classification_report(y_test, y_pred_dt,output_dict=True)
    print(classification_report(y_test, y_pred_dt))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_dt))
    cm = confusion_matrix(y_test, y_pred_dt)
    dt_cr = pd.DataFrame(dt_cr).transpose()
    dt_cr = pd.DataFrame(dt_cr)
    return dt_cr.to_html


def build_mlp():
    dt = MLPClassifier()
    dt.fit(X_train_dtm, y_train)
    y_pred_dt = dt.predict(X_test_dtm)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    print("Classification Accuracy:", dt_acc)
    print("Classification Report\n")
    dt_cr = classification_report(y_test, y_pred_dt,output_dict=True)
    print(classification_report(y_test, y_pred_dt))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_dt))
    cm = confusion_matrix(y_test, y_pred_dt)
    dt_cr = pd.DataFrame(dt_cr).transpose()
    dt_cr = pd.DataFrame(dt_cr)
    return dt_cr.to_html

def predict_userInput(posting):
    dt = DecisionTreeClassifier()
    dt.fit(X_train_dtm, y_train)
    # input_text = ["customer service associate us, ca, san francisco novitex enterprise solutions, formerly pitney bowes management services, delivers innovative document communications management solutions help companies around world drive business process efficiencies, increase productivity, reduce costs improve customer satisfaction. almost 30 years, clients turned us integrate optimize enterprise-wide business processes empower employees, increase productivity maximize results. trusted partner, continually focus delivering secure, technology-enabled document communications solutions improve clients' work processes, enhance customer interactions drive growth. customer service associate based san francisco, ca. right candidate integral part talented team, supporting continued growth.responsibilities:perform various mail center activities (sorting, metering, folding, inserting, delivery, pickup, etc.)lift heavy boxes, files paper neededmaintain highest levels customer care demonstrating friendly cooperative attitudedemonstrate flexibility satisfying customer demands high volume, production environmentconsistently adhere business procedure guidelinesadhere safety procedurestake direction supervisor site managermaintain logs reporting documentation; attention detailparticipate cross-training perform duties assigned (filing, outgoing shipments, etc)operating mailing, copy scanning equipmentshipping &amp; receivinghandle time-sensitive material like confidential, urgent packagesperform tasks assignedscanning incoming mail recipientsperform file purges pullscreate files ship filesprovide backfill neededenter information daily spreadsheetsidentify charges match billingsort deliver mail, small packages minimum requirements:minimum 6 months customer service related experience requiredhigh school diploma equivalent (ged) requiredpreferred qualifications:keyboarding windows environment pc skills required (word, excel powerpoint preferred)experience running mail posting equipment plusexcellent communication skills verbal writtenlifting 55 lbs without accommodationswillingness availability work additional hours assignedwillingness submit pre-employment drug screening criminal background checkability effectively work individually team environmentcompetency performing multiple functional tasksability meet employer's attendance policy computer software"]
    input_text = [posting]
    # convert text to feature vectors
    input_data_features = vect.transform(input_text)
    # making prediction
    prediction = dt.predict(input_data_features)
    print(prediction)
    if (prediction[0] == 'f'):
        result = 'Warning: Scam Job! Avoid at All Costs!'
    else:
        result = 'Authentic Posting. Best Wishes!'
    return result