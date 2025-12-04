import torch

from multiManagement.DatasetImageEncoder import DatasetImageEncoder
from multiManagement.TextTokenizer import TextTokenizer


class DatasetHelper:

    def __init__(self, num_words_to_keep):
        self.image_encoder = DatasetImageEncoder()
        self.tokenizer = TextTokenizer(num_words_to_keep)

    def preprocess_labels(self, train_data, val_data):
        if val_data == None:
            return train_data.get_labels()
        return train_data.get_labels(), val_data.get_labels()

    def preprocess_images(self, train_images, val_images):
        return self.images_encoder(train_images, val_images)

    def preprocess_texts(self, train_texts, val_texts, num_words_x_doc):
        self.train_tokenizer(train_texts)

        train_t = self.texts_to_indices(train_texts)
        for i in range(len(train_t)):
            if len(train_t[i]) > num_words_x_doc:
                train_t[i] = train_t[i][:num_words_x_doc]
            elif len(train_t[i]) < num_words_x_doc:
                train_t[i].extend([0] * (num_words_x_doc - len(train_t[i])))
        train_t = torch.tensor(train_t)
        if val_texts != None:
            val_t = self.texts_to_indices(val_texts)
            for i in range(len(val_t)):
                if len(val_t[i]) > num_words_x_doc:
                    val_t[i] = val_t[i][:num_words_x_doc]
                elif len(val_t[i]) < num_words_x_doc:
                    val_t[i].extend([0] * (num_words_x_doc - len(val_t[i])))
            val_t = torch.tensor(val_t)
            return train_t, val_t
        else:
            return train_t

    def images_encoder(self, train_images, val_images):
        return self.image_encoder.images_encoder(train_images, val_images)

    def train_tokenizer(self, train_texts):
        self.tokenizer.train_tokenizer(train_texts)

    def texts_to_indices(self, texts):
        return self.tokenizer.convert_to_indices(texts)


