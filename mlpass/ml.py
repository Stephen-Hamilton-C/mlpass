import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

VOCABULARY = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890~`!@#$%^&*()[]{}-_+=\\|;:'\"/?,.<>"
VOCAB_SIZE = max([ord(char) for char in VOCABULARY]) + 1

KAGGLE_DIR = Path("kaggle")
PWLDS_DIR = Path("pwlds")


def _split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.shuffle(data.values)

    data1 = data[0:int(len(data) / 2)]
    data2 = data[int(len(data) / 2):]

    df1 = pd.DataFrame(data1, columns=data.columns)
    df2 = pd.DataFrame(data2, columns=data.columns)
    return df1, df2


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading password datasets...")
    kaggle_data = pd.read_csv(KAGGLE_DIR.joinpath("data-processed.csv"))
    pwlds_very_weak = pd.read_csv(PWLDS_DIR.joinpath("pwlds_very_weak.csv"))
    pwlds_weak = pd.read_csv(PWLDS_DIR.joinpath("pwlds_weak.csv"))
    pwlds_avg = pd.read_csv(PWLDS_DIR.joinpath("pwlds_average.csv"))
    pwlds_strong = pd.read_csv(PWLDS_DIR.joinpath("pwlds_strong.csv"))
    pwlds_very_strong = pd.read_csv(PWLDS_DIR.joinpath("pwlds_very_strong.csv"))

    all_data = pd.concat([kaggle_data, pwlds_very_weak, pwlds_weak, pwlds_avg, pwlds_strong, pwlds_very_strong])
    # all_data = kaggle_data
    print("Data loaded. Splitting into training and test data...")

    train_data, test_data = _split_data(all_data)
    print("Split complete")
    return train_data, test_data


class Machine:
    def __init__(self, *, load_path: str = None, max_length=256):
        self.charCount = None
        self.truncated_password_flag = False
        self.illegal_char_flag = False
        self.load_path = load_path
        self.max_length = max_length

        if load_path is not None and os.path.isfile(load_path):
            self.model = load_model(load_path)
            self.loaded_from_file = True
        else:
            self._create_model()
            self.loaded_from_file = False

    def _create_model(self):
        self.model = Sequential()

        self.model.add(Embedding(VOCAB_SIZE, 16, input_length=self.max_length))
        self.model.add(LSTM(64))
        self.model.add(Dense(5, activation='softmax'))

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def _encode(self, password) -> np.array:
        chars = []
        for char in str(password):
            if char not in VOCABULARY:
                self.illegal_char_flag = True
                continue
            char_value = ord(char)
            chars.append(char_value)

        padding = [0] * (self.max_length - len(chars))
        chars.extend(padding)
        if len(chars) > self.max_length:
            self.truncated_password_flag = True
            chars = chars[:self.max_length]

        return np.array(chars)

    def save(self):
        model_path = self.load_path
        if self.load_path is None:
            num = 0
            os.makedirs("models", exist_ok=True)
            while os.path.exists(f"models/mlpass{num}.h5"):
                num += 1
            model_path = f"models/mlpass{num}.h5"

        if not model_path.endswith(".h5"):
            model_path += ".h5"

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def _prepare_data(self):
        self.train_data, self.test_data = load_data()
        print("Loaded datasets")
        self.train_x = np.array(self.train_data["Password"])
        self.train_y = np.array(self.train_data["Strength_Level"])
        self.test_x = np.array(self.test_data["Password"])
        self.test_y = np.array(self.test_data["Strength_Level"])
        print("Split datasets into training and test data")

        print("Encoding dataset...")
        print(self.train_x)
        print(self.test_x)
        self.train_x = np.array([self._encode(x) for x in self.train_x])
        self.test_x = np.array([self._encode(x) for x in self.test_x])
        print("Encoded dataset")

    def train(self, epochs: int):
        self._prepare_data()
        self.model.fit(
            x=self.train_x,
            y=self.train_y,
            validation_data=(self.test_x, self.test_y),
            shuffle=True,
            epochs=epochs
        )

    def validate(self) -> float:
        i = 0
        correct = 0
        for password in self.test_x:
            if self.predict(password) == self.test_y[i]:
                correct += 1
            i += 1

        # i at this point is now the total number of items in the dataset
        return correct / i

    def predict(self, password: str) -> int:
        return self.model.predict(self._encode(password))


if __name__ == "__main__":
    machine = Machine(load_path="test.h5", max_length=256)
    print("Training...")
    machine.train(5)
    print("Validating...")
    accuracy = machine.validate()
    print(f"Accuracy: {round(accuracy * 100, 2)}%")

    if machine.truncated_password_flag:
        print(
            f"WARNING: At least 1 password had to be truncated! Consider increasing max_length ({machine.max_length}).")
    if machine.illegal_char_flag:
        print(
            f"WARNING: At least 1 password used an illegal character! Consider updating your vocabulary ({VOCABULARY})")

    machine.save()
