#%%
import pandas as pd
import numpy as np
import sklearn
import kaggle

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from pathlib import Path
import joblib
import os

#%%
# store model version
sklearn_version = sklearn.__version__

# %% 
# download data from Kaggle in /datasets folder
dataset = 'ashishpatel26/sentimental-analysis-nlp'
file_name = "train_data.csv"
kaggle.api.dataset_download_file(dataset, file_name=file_name,
path=r'datasets')
#%% 
# import data
sentim_data = pd.read_csv("datasets/train_data.csv", names = ["label", "text"],sep = "\t")
sentim_data.sample(10) # inspect sample

# %% 
# assign features/target and ttsplit
X = sentim_data.text
y = sentim_data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% 
# instantiate tf-idf transformer obj
tfidf_vect = TfidfVectorizer(max_features=20)

# %%
# instantiate linearsvc model
lin_svc = LinearSVC(C=1., max_iter=1000, tol=1e-3)
#%%
# define pipeline
clf_pipeline = Pipeline(
    [('tfidf_vect', tfidf_vect),
    ('classifier', lin_svc)]
)
pipeline_model = clf_pipeline.fit(X_train, y_train)
# %%
# evaluate classifier
y_pred = pipeline_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# %%
# store job objects & metadata
text_clf_pipeline_param = {}
text_clf_pipeline_param['pipeline'] = pipeline_model
text_clf_pipeline_param['class labels'] = {0:"negative", 1: "positive"}
text_clf_pipeline_param['sklearn version'] = sklearn_version
text_clf_pipeline_param['test accuracy'] = test_accuracy

# %%
# create models dir if not present
Path(os.path.join(os.getcwd(),"models")).mkdir(parents=True, exist_ok=True)
# dump pipeline w metadata
out_filename = "models/pipe_clf_checkpoint.joblib"
joblib.dump(text_clf_pipeline_param, out_filename)

