from sklearn.base import BaseEstimator,TransformerMixin
import re
import string
class TextCleaner(BaseEstimator,TransformerMixin):
    def remove_mentions(self,input_text):
        return re.sub('@\S+','',input_text)
    def remove_urls(self,input_text):
        return re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+','',input_text)
    def remove_punctuation(self,input_text):
        punct = string.punctuation
        punct = punct.replace("'","")
        trantab = str.maketrans(punct,len(punct)*' ')
        return input_text.translate(trantab)
    def remove_digits(self,input_text):
        return re.sub('\d+','',input_text)
    def to_lower(self,input_text):
        return input_text.lower()
    def resolve_hashtags(self,text):
        return text.replace("#","")
    def remove_3consecutive(self,text):
        return re.sub(r"([a-zA-Z0-9])\1\1+", r"\1", text)
    def check_empty(self,text):
        if text ==" ":
            return None
        else:
            return text
    def fit(self,X,y=None,**fit_params):
        return self
    def transform(self,X,**transform_params):
        clean_X = X.apply(self.remove_urls).apply(self.remove_mentions).apply(self.resolve_hashtags).apply(self.remove_punctuation).apply(self.to_lower).apply(self.remove_digits).apply(self.remove_3consecutive).apply(self.check_empty)
        return clean_X
tc = TextCleaner()