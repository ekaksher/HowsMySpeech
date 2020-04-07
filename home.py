import flask
import pickle
import pandas as pd
from nltk.stem import PorterStemmer
from flask_sqlalchemy import SQLAlchemy
app = flask.Flask(__name__,template_folder='templates',static_folder='static')
model = pickle.load(open("model/sentiment_tfifbi100k.pkl","rb"))
from cleaner import tc
ENV = 'prod'
if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://ekaksher:ekhere@localhost/'
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://oquswbzduzkfoa:af4ecc5afdb18adfb8590c9222828fe964e51603fd532b055435349c4ee77158@ec2-18-235-97-230.compute-1.amazonaws.com:5432/db58boe0rb5u01'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
porter = PorterStemmer()
def stemwords(input_text):
    words = input_text[0].split()
    text = " ".join([porter.stem(word) for word in words])
    return text
class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer,primary_key=True)
    rating = db.Column(db.Integer)
    def __init__(self,rating):
        self.rating = rating
@app.route("/",methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        inputtext = flask.request.form['text']
        text = pd.Series(inputtext)
        text = tc.fit_transform(text)
        text = stemwords(text)
        prob = model.predict_proba(pd.Series(text))
        prediction = [1 if prob[0][1]>0.51 else 0]
        if prediction == [0]:
            prediction = "Negative"
            confidence = int(prob[0][0]*100)
            color = "red"
        else:
            prediction = "Positive"
            confidence = int(prob[0][1]*100)
            color = "green"
        return flask.render_template('main.html',original_input=inputtext,result=prediction,colors=color,confidence=confidence)
@app.route("/feedback",methods=['POST'])
def feedback():
    if flask.request.method == 'POST':
        data  = flask.request.form['slider']
        data = Feedback(int(data))
        db.session.add(data)
        db.session.commit()
        return(flask.render_template('main.html'))
if __name__=="__main__":
    app.run()
