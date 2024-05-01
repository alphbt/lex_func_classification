from transformers import  AdamW, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from BertDataset import BertDataset

class BertClassifier:
    
    def __init__(self,
                 model_path,
                 tokenizer_path,
                 n_classes,
                 label_encoder,
                 max_len,
                 epochs,
                 batch_size,
                 model_save_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size

        self.label_encoder = label_encoder

        self.model_save_path = model_save_path

        self.model.to(self.device)
    
    def prepare(self, X_train, y_train, X_val, y_val):
        self.train_set = BertDataset(pd.concat([X_train, y_train], axis=1),
                                     self.tokenizer,
                                     self.label_encoder,
                                     self.max_len)
        self.validation_set = BertDataset(pd.concat([X_val, y_val], axis=1),
                                     self.tokenizer,
                                     self.label_encoder,
                                     self.max_len)
        
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.validation_loader = DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.train_loader) * self.epochs)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    
    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for bacth_idx, data in enumerate(self.train_loader):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
            targets = data['targets'].to(self.device, dtype = torch.long)

            outputs = self.model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
            
            predictions = torch.argmax(outputs.logits, dim=1)
            loss =self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(predictions == targets)
            losses.append(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        
        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss
    
    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(self.validation_loader):
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.long)

                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.validation_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss
    
    def train(self):
        best_accuracy = 0

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')

            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

            self.model = torch.load(self.model_save_path)
            self.model.to(self.device)
    
    def predict(self, sentence, collocation):
        encoding = self.tokenizer.encode_plus(
            sentence,
            collocation.lower(),
            add_special_tokens = True,
            max_length = self.max_len,
            return_token_type_ids = True,
            return_attention_mask = True,
            return_tensors = 'pt',
            padding = 'max_length'
        )

        out = {
            'text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding["token_type_ids"].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        token_type_ids = out["token_type_ids"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0),
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(prediction)[0]