import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoModelForSequenceClassification, BertTokenizerFast

# specify GPU
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")


def handle_tokenize(text, labels):
    max_seq_len = 100
    # print('starting tokenizing...')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # text should be list
    encoding = tokenizer.batch_encode_plus(text, padding=True,
                                           truncation=True, max_length=max_seq_len,
                                           return_token_type_ids=False)  # todo what is the default max_len? double check the params

    # for train set
    # print(encoding['input_ids'].type())
    seq = torch.tensor(encoding['input_ids'])
    mask = torch.tensor(encoding['attention_mask'])  # todo what does this mask do?
    # y = torch.tensor(labels)
    # print('seq, mask and labels are ready')

    return seq, mask


if __name__ == '__main__':
    # load weights of best model
    path = 'saved_weights_e10_b6.pt'
    config = AutoConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, return_dict=True)
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load(path))
    model.to(device)

    with open('new_splits/test_labels.csv') as f:
        labels = f.read().splitlines()

    labels = [int(x) for x in labels]

    with open('new_splits/test_text.csv') as f:
        texts = f.read().splitlines()

    print('num of test articles', len(texts))
    if len(texts) != len(labels):
        print('ERRROOOOR!')

    step = 30
    all_preds = []
    embs = []
    for i in range(0, len(texts), step):
        print('i', i)
        if i + step < len(labels):
            sub_texts = texts[i:i + step]
            sub_labels = labels[i:i + step]
        else:
            sub_texts = texts[i:]
            sub_labels = labels[i:]

        test_seq, test_mask = handle_tokenize(sub_texts, sub_labels)



        # get predictions for test data
        # print('going to get preds from model')
        with torch.no_grad():
            outputs = model(test_seq.to(device), test_mask.to(device))
            # print(outputs.last_hidden_state)
            # print(outputs.last_hidden_state.size())
            h = outputs.hidden_states
            preds = outputs.logits
            # print(preds)
            preds = preds.detach().cpu().numpy()
            test_seq.detach().cpu()
            test_mask.detach().cpu()

        # print(h[0][:, 0, :].size())
        e = torch.zeros(h[0][:, 0, :].size()).to(device)
        # print(type(e))
        for layer in range(len(h)-1):
            # print('hiddens ', layer, h[layer].size())
            e += h[layer][:, 0, :]

          # todo which index to take? 0 or -1
        # print(e.size())
        e = e.detach().cpu().numpy()
        # print('e shape', e.shape)
        embs.append(e)
        preds = np.argmax(preds, axis=1)
        all_preds.append(preds)

    embeddings = np.vstack(embs)

    print('embeddings shape', embeddings.shape)
    np.savetxt('fine_tuned_embeddings_e10_b6_avg.csv', embeddings, delimiter=',')
    all_preds = np.hstack(all_preds)
    print(classification_report(labels, all_preds))
