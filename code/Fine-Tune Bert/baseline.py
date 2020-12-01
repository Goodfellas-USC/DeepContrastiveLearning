# summary 1e-5 max_length 500
import numpy as np
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, BertTokenizer  # todo fast?

from plots import multi_plot_hist

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
# specify GPU
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") # todo change to cuda later

def len_text(text):
    if text.split() > 200:
        return True
    return False

def read_raw_data(data_path):
    if not data_path:
        print('reading prepared data')
        with open('new_splits/train_text.csv') as f:
            content = f.readlines()
            train_text = [x.strip() for x in content]
        with open('new_splits/val_text.csv') as f:
            content = f.readlines()
            val_text = [x.strip() for x in content]

        with open('new_splits/train_labels.csv') as f:
            content = f.readlines()
            train_labels = [int(x.strip()) for x in content]
        with open('new_splits/val_labels.csv') as f:
            content = f.readlines()
            val_labels = [int(x.strip()) for x in content]

        return train_text, train_labels, val_text, val_labels

    label_col = 'label'
    text_col = 'text'
    df = pd.read_csv(data_path)
    # min_len = 200
    # df = df[df[text_col].str.split().str.len().gt(min_len)]
    print('sources: ', df.source.unique())
    df['label'] = df['source'].map({'Breitbart': 1, 'Huffington Post US': 0})

    train_text, temp_text, train_labels, temp_labels = train_test_split(df[text_col], df[label_col],
                                                                        random_state=2020,
                                                                        test_size=0.3,
                                                                        stratify=df[label_col])


    # we will use temp_text and temp_labels to create validation and test set
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=2020,
                                                                    test_size=0.5,
                                                                    stratify=temp_labels)
    # print(sum(train_labels), train_labels.shape)
    # train_size(9963, )
    # val_size(2135, )
    return train_text, train_labels, val_text, val_labels, test_text, test_labels


def handle_data(batch_size, df_path=None):
    train_text, train_labels, val_text, val_labels = read_raw_data(df_path)
    # todo do indices make problem?

    train_data = handle_tokenize(train_text, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    print('trainloader ready')

    val_data = handle_tokenize(val_text, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
    print('valloader ready')

    # test_data = handle_tokenize(test_text, test_labels)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # print('testloader ready')
    return train_dataloader, val_dataloader#, test_dataloader


def handle_tokenize(text, labels):
    max_seq_len = 200

    print('starting tokenizing...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # encoding = tokenizer.batch_encode_plus(text.tolist(), padding=True,
    #                                        truncation=True, max_length=max_seq_len,
    #                                        return_token_type_ids=False)  # todo what is the default max_len? double check the params

    encoding = tokenizer(text, max_length=max_seq_len, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False)# dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    seq = encoding['input_ids'] # default max_seq 512
    # print('*********** seq size:', seq.size())
    mask = encoding['attention_mask']  # todo what does this mask do?
    y = torch.Tensor(labels)
    print('seq, mask and labels are ready')

    return TensorDataset(seq, mask, y)


def train_an_epoch(train_dataloader, model):
    model.train()  # TODO set to false if evaluating

    total_loss = 0
    all_loss =[]
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # cross_entropy = nn.CrossEntropyLoss()  # handles Softmax inside of it

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()
        optimizer.zero_grad()

        labels = labels.long().unsqueeze(0)
        outputs = model(input_ids=sent_id, attention_mask=mask, labels=labels) # 'loss', 'logits', 'hidden_states'
        loss = outputs.loss
        # logits = outputs.logits

        loss.backward()
        optimizer.step()
        all_loss.append(loss)
        total_loss = total_loss + loss.item()

        # preds  = logits
        # preds = preds.detach().cpu().numpy()
        # total_preds.append(preds)
        for r in batch:
            r.detach().cpu()

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # total_preds = np.concatenate(total_preds, axis=0)


    return avg_loss, all_loss # todo is it ok that I am returning model?


def evaluate(dataloader, model):
    print("\nEvaluating...")

    model.eval() # deactivate dropout layers
    total_loss = 0
    total_num = 0
    correct = 0

    for step, batch in enumerate(dataloader):

        if step % 300 == 0 and not step == 0:
            print('Evaluating  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            labels = labels.long().unsqueeze(0)
            # print('labels size', labels.size())
            outputs = model(input_ids=sent_id, attention_mask=mask, labels=labels)  # 'loss', 'logits', 'hidden_states'
            loss = outputs.loss
            total_loss = total_loss + loss.item()

            logits = outputs.logits

            _, predicted = torch.max(logits.data, 1)
            total_num += labels.size(1)
            correct += (predicted == labels).sum().item()
        batch = [t.detach().cpu for t in batch]

    avg_loss = total_loss / len(dataloader)
    acc = correct / total_num


    return avg_loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net arguments.')

    parser.add_argument('-e', type=str, help='num of epochs')
    parser.add_argument('-b', type=str, help='batch size')

    args = parser.parse_args()

    epochs = int(args.e)
    batch_size = int(args.b)

    print('GPU details:', torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.is_available())

    # df_path = './aylien_huff_breit_14234.csv'

    train_data_loader, val_data_loader = handle_data(batch_size=batch_size)

    config = AutoConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, return_dict=True)
    model = AutoModelForSequenceClassification.from_config(config)

    model = model.to(device)  # push the model to GPU

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    model_path = f'saved_weights_e{epochs}_b{batch_size}.pt'

    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        _, all_loss = train_an_epoch(train_data_loader, model)

        valid_loss, valid_acc = evaluate(val_data_loader, model)

        # save the best model
        if valid_loss < best_valid_loss:
            print(f'best weights updated in epoch: {epoch}')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)


        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'Validation acc: {valid_acc:.3f}')

        # todo calculate accuracy of model here
        train_loss, train_acc = evaluate(train_data_loader, model)
        print(f'Training Loss: {train_loss:.3f}')
        print(f'Training acc: {train_acc:.3f}')

        train_losses.extend(all_loss)
        # valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

    multi_plot_hist(train_losses, valid_losses, train_accs, valid_accs, 'learning_curve')
    # test_measure(train_data_loader, model, model_path, train_y)
    # test_measure(test_data_loader, model, model_path, test_y)

