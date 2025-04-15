
from pymongo import MongoClient
from pymongo.server_api import ServerApi

def get_dbase():
    uri = "mongodb+srv://cycyBell:Cycy=789@cluster0.0lzbxfo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client['Data_Scientists']
    except Exception as e:
        print(e)


        