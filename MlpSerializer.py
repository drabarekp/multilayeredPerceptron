import pickle


class MlpSerializer:

    def __init__(self):
        self.root_path = "saved_networks/"

    def serialize(self, network, filename):
        path = self.root_path + filename
        with open(path, "wb") as outfile:
            pickle.dump(network, outfile)

    def deserialize(self, filename):
        path = self.root_path + filename
        with open(path, "rb") as infile:
            test_dict_reconstructed = pickle.load(infile)
        return test_dict_reconstructed