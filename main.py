from pymongo.mongo_client import MongoClient

from load import clothes
from video_capture import emotiv

uri = "mongodb+srv://Cuza:FvcsS3QJWzhGKRkk@cuza.yuiityt.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
def inser_data(data):
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
if __name__ == '__main__':
    try:
        client.admin.command('ping')
        print('Connected successfully!')
        db = client.sample_guides
        coll = db.data
        while True:
            x=read_Data("sample_guides","comets")
            if x!=None:
                print("There is smt {x}")
                if x['code']==2:
                    print("emotions")
                    a=emotiv()
                    delete_data("sample_guides","comets")
                elif x['code']==1:
                    a=clothes()
                    inser_data([{"clothes":a}])
                    delete_data("sample_guides","comets")

    except Exception as e:
        print(e)
