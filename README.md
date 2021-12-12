# flask_sklearn_model_deployment
Simple sklearn LinearSVC model training and deployment with Flask

Example training data is downloaded in a datasets folder via the Kaggle API (make sure to install the kaggle package and store your API token in the .json folder; if unsure [read the Kaggle API docs](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication))

Following training, the model & its metadata is stored in the models folder using joblib.

Test the api by running requests.py (in a new terminal) after running the Flask app server.

To edit/add requests, simply edit the "input" field in the api_test_data.json file.
It is possible to ask the api for both 0/1 or negative/positive class labels when making predictions by editing the "class labels" filed of the request (set to "True" for labels).  
