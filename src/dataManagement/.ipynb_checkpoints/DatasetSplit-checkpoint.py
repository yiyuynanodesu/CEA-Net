class DatasetSplit(object):

    def __init__(self):
        self.labels = []
        self.images = []
        self.texts = []

    def load_data(self, data):
        for line in data:
            img_path,describe,label = line
            self.labels.append(label)
            self.images.append(img_path)
            self.texts.append(describe)

    def get_labels(self):
        return self.labels

    def get_images(self):
        return self.images

    def get_texts(self):
        return self.texts

    def set_labels(self, labels):
        self.labels = labels

    def set_images(self, images):
        self.images = images

    def set_texts(self, texts):
        self.texts = texts
