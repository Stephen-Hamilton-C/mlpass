import os
from collections.abc import Iterable
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

VOCABULARY = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890~`!@#$%^&*()[]{}-_+=\\|;:'\"/?,.<> "
VOCAB_SIZE = max([ord(char) for char in VOCABULARY]) + 1

KAGGLE_DIR = Path("kaggle")
PWLDS_DIR = Path("pwlds")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")


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
        self.loaded_from_file = False
        self.truncated_password_flag = False
        self.illegal_char_flag = False
        self.load_path = load_path
        self.max_length = max_length

        if load_path is not None:
            model_path = MODELS_DIR.joinpath(load_path)
            if model_path.suffix != ".keras":
                model_path = model_path.with_suffix(".keras")

            if model_path.is_file():
                self.model = load_model(model_path)
                self.loaded_from_file = True

        if not self.loaded_from_file:
            self._create_model()
            self.loaded_from_file = False

    def _create_model(self):
        self.model = Sequential()

        self.model.add(Embedding(VOCAB_SIZE, 16, input_length=self.max_length))
        self.model.add(LSTM(64))
        self.model.add(Dense(5, activation='softmax'))

        self.model.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def _encode(self, password) -> np.array:
        chars = []
        for char in str(password):
            if char not in VOCABULARY:
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
        try:
            model_path = MODELS_DIR.joinpath(self.load_path)
            if self.load_path is None:
                num = 0
                os.makedirs(MODELS_DIR, exist_ok=True)
                while MODELS_DIR.joinpath(f"mlpass{num}.keras").exists():
                    num += 1
                model_path = MODELS_DIR.joinpath(f"mlpass{num}.keras")

            if model_path.suffix != ".keras":
                model_path.with_suffix(".keras")

            os.makedirs(str(model_path.parent), exist_ok=True)
            self.model.save(model_path)
        except IOError:
            print("Failed to save model!")

    def _prepare_data(self):
        train_x_path = DATA_DIR.joinpath("train_x.npy")
        train_y_path = DATA_DIR.joinpath("train_y.npy")
        test_x_path = DATA_DIR.joinpath("test_x.npy")
        test_y_path = DATA_DIR.joinpath("test_y.npy")
        if train_x_path.is_file() and train_y_path.is_file() and test_x_path.is_file() and test_y_path.is_file():
            print("Found previously encoded dataset. Loading from file...")
            self.train_x = np.load(train_x_path)
            self.train_y = np.load(train_y_path)
            self.test_x = np.load(test_x_path)
            self.test_y = np.load(test_y_path)
            return

        self.train_data, self.test_data = load_data()
        print("Loaded datasets")
        self.train_x = np.array(self.train_data["Password"])
        self.train_y = np.array(self.train_data["Strength_Level"])
        self.test_x = np.array(self.test_data["Password"])
        self.test_y = np.array(self.test_data["Strength_Level"])
        print("Split datasets into training and test data")

        print("Encoding dataset...")
        self.train_x = np.array([self._encode(x) for x in self.train_x])
        self.test_x = np.array([self._encode(x) for x in self.test_x])
        print("Saving dataset to file...")
        try:
            np.save(train_x_path, self.train_x)
            np.save(train_y_path, self.train_y)
            np.save(test_x_path, self.test_x)
            np.save(test_y_path, self.test_y)
        except IOError:
            print("Failed to save dataset to file. Continuing...")

    def train(self, epochs: int):
        if self.loaded_from_file:
            print("This model has already been loaded from a file! Skipping the training step...")
            return

        self._prepare_data()
        print(self.train_x, self.train_y, self.test_x, self.test_y)
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

    def predict(self, passwords: Iterable[str]) -> int:
        encoded_passwords = np.array([self._encode(pwd) for pwd in passwords])
        category_predictions = self.model.predict(encoded_passwords)
        predictions = []
        print(category_predictions)
        for cat_pred in category_predictions:
            max_pred = 0
            max_pred_i = 0
            for i, pred in enumerate(cat_pred):
                if pred > max_pred:
                    max_pred = pred
                    max_pred_i = i
            predictions.append(max_pred_i)
        return predictions


if __name__ == "__main__":
    # Problem with this approach: The algorithm thinks passwords with more null characters are better, rather than ignoring null characters.
    # Perhaps we should try tokenization again...
    machine = Machine(load_path="sparse-cat-cross_rmsprop_5.keras", max_length=256)
    print("Training...")
    machine.train(5)
    print("Saving model to file...")
    machine.save()
    print("Predict test...")
    print(machine.predict(["1234", "Password", "Th!$ i$ my SUPER ultr@ m3g5 CR4ZY P5SSW0R8 64"]))
    # print("Validating...")
    # accuracy = machine.validate()
    # print(f"Accuracy: {round(accuracy * 100, 2)}%")

    if machine.truncated_password_flag:
        print(
            f"WARNING: At least 1 password had to be truncated! Consider increasing max_length ({machine.max_length}).")
    if machine.illegal_char_flag:
        print(
            f"WARNING: At least 1 password used an illegal character! Consider updating your vocabulary ({VOCABULARY})")

