from flask import Flask,request,redirect,url_for,render_template
from form import textForm
from flask_bootstrap import Bootstrap
import joblib
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from flask_nav import Nav
from flask_nav.elements import Navbar, View
df=pd.read_csv('tweet-sentiment-extraction/train.csv')
df.loc[8].selected_text
df.fillna(value='none',inplace=True)
df['text'] = df['text'].str.replace('\d+', '')
df['text']=df['text'].str.replace('_',' ')
X_train, X_test, y_train, y_test = train_test_split(df.text, df.sentiment, random_state=1)
le = preprocessing.LabelEncoder()
#vect = CountVectorizer(max_features=2900,min_df=4)
vect=TfidfVectorizer(min_df=4,max_features=4000)
y_test=le.fit_transform(y_test)
y_train=le.fit_transform(y_train)
X_train_dmt=vect.fit_transform(X_train)
X_test_dmt=vect.transform(X_test)


with open ('static/model.sav','rb') as f:
    clf=joblib.load(f)


app = Flask(__name__)
Bootstrap(app)
app.secret_key="pass"
nav = Nav()

@nav.navigation()
def mynavbar():
    return Navbar(
        'mysite',
        View('home', 'index'),
    )

@app.route('/', methods=['GET', 'POST'])
def home():
    if (request.method=='POST'):
        
        ans=le.inverse_transform(clf.predict(vect.transform([request.form["player"]])))
        return render_template('home.html', ans=ans[0])
    return render_template('home.html')
nav.init_app(app)


if __name__=="__main__":
    app.run(debug=True)