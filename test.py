from pymongo.mongo_client import MongoClient
from playsound import playsound
uri = "mongodb+srv://Cuza:FvcsS3QJWzhGKRkk@cuza.yuiityt.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
def inser_data(data):
    db = client.sample_guides
    coll = db.comets
    result = coll.insert_many(data)
def read_Data(database_name, collection_name):
    db = client[database_name]
    col = db[collection_name]
    # Collection Name
    x = col.find_one()
    return x
def delete_data(database_name, collection_name):
    mydb = client[database_name]
    mycol = mydb[collection_name]

    x = mycol.delete_many({})

    print(x.deleted_count, " documents deleted.")
def clothes():
    try:
        client.admin.command('ping')
        print('Connected successfully!')
        db = client.sample_guides
        coll = db.comets
        inser_data([{"code": 1}] )

    except Exception as e:
        print(e)
def emotions():
    try:
        client.admin.command('ping')
        print('Connected successfully!')
        db = client.sample_guides
        coll = db.comets
        inser_data([{"code": 2}] )

    except Exception as e:
        print(e)
def emotions():
     try:
        client.admin.command('ping')
        print('Connected successfully!')
        inser_data([{"code":2}])
        while True:
            x=read_Data("sample_guides","data")
            if x!=None:
                if x['emotiom']=="Happy":
                    delete_data("sample_guides","data")
                    delete_data("sample_guides","comets")
                    return "You are doing well!"

                elif x['emotiom']=="Angry":
                    delete_data("sample_guides","data")
                    delete_data("sample_guides","comets")
                    return "CAlm down!"
                elif x['emotiom']=="Sad":
                    delete_data("sample_guides","data")
                    delete_data("sample_guides","comets")
                    return "there will be better times"

     except Exception as e:
        print(e)
def clothes():
    try:
        client.admin.command('ping')
        print('Connected successfully!')
        inser_data([{"code": 1}])
        while True:
            x = read_Data("sample_guides", "data")
            if x != None:
                delete_data("sample_guides", "data")
                delete_data("sample_guides", "comets")
                return x['clothes']

    except Exception as e:
        print(e)