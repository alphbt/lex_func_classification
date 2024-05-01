from torch.utils.data import Dataset
import torch

class BertDataset(Dataset):
    def __init__(self, 
                 data, 
                 tokenizer, 
                 label_encoder, 
                 max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentences = self.data['sentence'].values
        self.collocations = self.data['collocation'].values
        self.targets = label_encoder.transform(self.data['function'].values)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        collocation = str(self.collocations[index])

        # [CLS] sentence [SEP] collocation
        inputs = self.tokenizer.encode_plus(
            sentence,
            collocation.lower(),
            add_special_tokens = True,
            max_length = self.max_len,
            return_token_type_ids = True,
            return_attention_mask = True,
            return_tensors = 'pt',
            padding = 'max_length'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.tensor(self.targets[index])
        }