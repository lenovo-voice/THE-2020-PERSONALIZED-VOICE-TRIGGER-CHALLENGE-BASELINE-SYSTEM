import pickle

def save_pickle(name, mydict):
    output = open(name, 'wb')
    pickle.dump(mydict, output)
    output.close()

def read_pickle(name):
    pkl_file = open(name, 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    return mydict

