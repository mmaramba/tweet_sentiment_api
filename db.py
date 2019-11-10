import pymongo
from credentials import *

username = get_username()
password = get_password()
connection_string = "mongodb+srv://{}:{}@cluster0-bdxdg.azure.mongodb.net/test?retryWrites=true&w=majority".format(username, password)
client = pymongo.MongoClient(connection_string)
my_db = client['test']

my_col = my_db['candidates']

my_dict = {'name': 'joe-biden', 'sentiment': 0, 'time': '9-23-19.8:00PM'}
x = my_col.insert_one(my_dict)

print(x.inserted_id)