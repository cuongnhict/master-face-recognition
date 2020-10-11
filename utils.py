import pickle


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
