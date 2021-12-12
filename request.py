import requests
import json

# read json file w api requests
with open("api_test_data.json") as f:
    my_requests = json.load(f)

# Flask app api
url = "http://localhost:8080/api"
r = requests.post(url, json= my_requests)

# print response
for resp in r.json(): print(resp)