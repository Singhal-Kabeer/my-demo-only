import pickle

def load_data():
    with open('../train_windowed.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('../dev_windowed.pkl', 'rb') as f:
        dev = pickle.load(f)
    with open('../test_windowed.pkl', 'rb') as f:
        test = pickle.load(f)

    return train, dev, test