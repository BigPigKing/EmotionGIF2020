import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

train_X_PATH = "train_X.npy"
train_Y_PATH = "train_Y.npy"
test_X_PATH = "test_X.npy"
test_Y_PATH = "test_Y.npy"
dev_X_PATH = "dev_X.npy"
dev_json_path = "data/dev_unlabeled.json"
categories_PATH = "data/categories.json"


def load_data(train_paths, test_paths, dev_path):
    train_X, train_Y = train_paths
    test_X, test_Y = test_paths
    dev_X = dev_path

    train_X = np.load(train_X, allow_pickle=True)
    train_Y = np.load(train_Y, allow_pickle=True)
    test_X = np.load(test_X, allow_pickle=True)
    test_Y = np.load(test_Y, allow_pickle=True)
    dev_X = np.load(dev_X, allow_pickle=True)

    return train_X, train_Y, test_X, test_Y, dev_X


def transform_to_tensor(input_array):
    feature_shape_list = []

    # get feature shape list for dynamic using
    for feature in input_array[0]:
        feature_shape_list.append(feature.shape[0])

    # transform to ideal shape
    ideal_list = []
    for row_idx, row in enumerate(input_array):
        feature_list = []

        for feature in row:
            feature_list.append(feature)

        all_features = np.concatenate(feature_list)
        ideal_list.append(all_features)
    ideal_array = np.vstack(ideal_list)

    return torch.from_numpy(ideal_array), feature_shape_list


def transform_to_submit_format(dev_path, pred_Y, category_list):
    dev_X_df = pd.read_json(dev_path, lines=True)
    pred_Y = pred_Y.cpu()
    name_pred_Y = []

    for pred_y in pred_Y:
        pred_cates = np.argsort(pred_y).tolist()[::-1][:6]
        name_pred_y_cates = []

        for cate_idx in pred_cates:
            name_pred_y_cates.append(category_list[cate_idx])

        name_pred_Y.append(name_pred_y_cates)

    dev_X_df.insert(1, "categories", name_pred_Y, True)
    dev_X_df.to_json("dev.json", orient='records', force_ascii=False, lines=True)


class Classify_Model(nn.Module):
    def __init__(self, e_feature_shape_list, output_dim):
        super(Classify_Model, self).__init__()

        self.e_feature_shape_list = e_feature_shape_list
        self.e_num = int(len(e_feature_shape_list)/2)
        self.e_linear_layers = nn.ModuleList(
            [nn.Linear(feature_shape, 512) for feature_shape in e_feature_shape_list]
        )

        self.e_combined_layers = nn.ModuleList([nn.Linear(1024, 512)] * self.e_num)
        self.final_layer = nn.Sequential(
            nn.Linear(512*self.e_num, 256*self.e_num),
            nn.ReLU(),
            nn.Linear(256*self.e_num, 128*self.e_num),
            nn.ReLU(),
            nn.Linear(128*self.e_num, output_dim)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0006)

    def forward(self, X):
        # step one output
        first_output_list = []
        start_feature = 0
        for e_feature_idx, e_feature_shape in enumerate(self.e_feature_shape_list):
            output = self.e_linear_layers[e_feature_idx](X[:, start_feature:start_feature+e_feature_shape])
            first_output_list.append(output)

            start_feature += e_feature_shape

        # step two output
        second_output_list = []
        e_idx = 0
        start = 0
        end = 1

        while end < len(first_output_list):
            e_combined = torch.cat((first_output_list[start], first_output_list[end]), 1)
            output = self.e_combined_layers[e_idx](e_combined)
            second_output_list.append(output)

            start += 2
            end += 2
            e_idx += 1

        # final output
        combined_output = torch.cat(second_output_list, 1)

        Y = self.final_layer(combined_output)

        return Y

    def optimize(self, batch_X, batch_Y):
        pred_Y = self(batch_X.float())

        batch_loss = self.criterion(pred_Y.float(), batch_Y.float())

        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_loss.item()

    def fit(self, train_X, train_Y, test_X, test_Y, epochs, batch_size):
        train = torch.utils.data.TensorDataset(train_X, train_Y)

        for epoch in range(epochs):
            train_batch_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size,
                                                             shuffle=True, num_workers=4)

            total_loss = 0.0
            for batch_X, batch_Y in train_batch_loader:
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_Y = batch_Y.cuda()
                batch_loss = self.optimize(batch_X, batch_Y)
                total_loss += batch_loss

            print("Epoch {}: Loss {:4f}".format(epoch+1, total_loss))
            self.predict_emotion(train_X, train_Y)
            self.predict_emotion(test_X, test_Y)

        self.save_param()

    def predict(self, input_X, batch_size):
        input_X_loader = torch.utils.data.DataLoader(dataset=input_X, batch_size=batch_size,
                                                     shuffle=False, num_workers=4)
        pred_Y = []
        for batch_X in input_X_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()

            with torch.no_grad():
                pred_batch_Y = self(batch_X.float())

            pred_Y.append(pred_batch_Y)

        pred_Y = torch.cat(pred_Y, 0)
        return pred_Y

    def predict_emotion(self, input_X, input_Y):
        df = pd.read_json(categories_PATH)
        emotion_list = df[0].to_list()

        with torch.no_grad():
            pred_Y = self.predict(input_X, 512)

        pred_Y = pred_Y.cpu()
        input_Y = input_Y.cpu()
        pred_Y = pred_Y.numpy()
        input_Y = input_Y.numpy()

        pred_emotion_record, acc_record = [], []
        for i in range(len(pred_Y)):
            denominator = np.sum(input_Y[i])
            answer = input_Y[i]

            max_idx = pred_Y[i].argsort()[-6:]
            num = np.sum([1 for idx in max_idx if answer[idx] == 1])
            acc = num/denominator
            acc_record.append(acc)

            emotion = [emotion_list[idx] for idx in max_idx]
            pred_emotion_record.append(emotion)

        acc = np.mean(acc_record)
        print("    Acc  |{:4f}".format(acc))
        # return pred_emotion, acc

    def save_param(self):
        torch.save(self.state_dict(), 'final_params.pkl')

    def load_param(self):
        self.load_state_dict(torch.load('final_params.pkl', map_location=torch.device('cpu')))


def main():
    train_X, train_Y, test_X, test_Y, dev_X = load_data((train_X_PATH, train_Y_PATH),
                                                        (test_X_PATH, test_Y_PATH),
                                                        dev_X_PATH)

    train_X, feature_shape_list = transform_to_tensor(train_X)
    train_Y, _ = transform_to_tensor(train_Y)
    test_X, _ = transform_to_tensor(test_X)
    test_Y, _ = transform_to_tensor(test_Y)
    dev_X, _ = transform_to_tensor(dev_X)

    classify_model = Classify_Model(feature_shape_list, 43)

    if torch.cuda.is_available():
        classify_model = classify_model.cuda()
    else:
        pass

    classify_model.fit(train_X, train_Y, test_X, test_Y, 30, 512)
    pred_Y = classify_model.predict(dev_X, 512)

    transform_to_submit_format(dev_json_path, pred_Y, pd.read_json(categories_PATH)[0].tolist())


if __name__ == '__main__':
    main()
