from jinja2 import Template
from ollama import Client
import re
import json

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn

from multi_aspect_classifier import MultiAspectClassifier

class PLMClassifier:

    def __init__(self, n_test, n_train, model_name: str, device: int):
        """
        Initialise le modèle, le tokenizer et les configurations nécessaires.
        """
        self.device = torch.device(f'cuda:{device}' if device >= 0 else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MultiAspectClassifier(model_name).to(self.device)
        self.n_test = n_test
        self.n_train = n_train
        print(f"n_test : {n_test}")
        print(f"n_train : {n_train}")
        

    def prepare_data(self, data: list[dict], max_length: int = 512, is_train: bool = True):
        texts = [item["Avis"] for item in data]
        labels = {
            aspect: [self.map_label(item[aspect]) for item in data]
            for aspect in ["Prix", "Cuisine", "Service", "Ambiance"]
        }
        
        if self.n_train > 0 and is_train:
            texts = texts[:self.n_train]
            for aspect in labels:
                labels[aspect] = labels[aspect][:self.n_train]
                              
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        labels = {aspect: torch.tensor(values) for aspect, values in labels.items()}
       
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"], encodings["attention_mask"],
            *[labels[aspect] for aspect in labels]
        )
        return dataset

    @staticmethod
    def map_label(label: str) -> int:
        """
        Mappe une étiquette texte vers un index numérique.
        """
        label_map = {"Positive": 0, "Négative": 1, "Neutre": 2, "NE": 3}
        return label_map[label]

    @staticmethod
    def inverse_map_label(index: int) -> str:
        """
        Mappe un index numérique vers une étiquette texte.
        """
        label_map = {0: "Positive", 1: "Négative", 2: "Neutre", 3: "NE"}
        return label_map[index]

    def train(self, train_data: list[dict], val_data: list[dict], batch_size: int, epochs: int, lr: float):
        """
        Entraîne le modèle sur les données d'entraînement et valide à chaque époque.
        :param train_data: Liste des dictionnaires avec "Avis" et "Aspect".
        :param val_data: Liste des dictionnaires pour validation.
        :param batch_size: Taille des lots.
        :param epochs: Nombre d'époques.
        :param lr: Taux d'apprentissage.
        """
        # Préparation des données d'entraînement
        train_dataset = self.prepare_data(data=train_data, is_train=True)
        val_dataset = self.prepare_data(data=val_data, is_train=False)
        max_val_size = self.n_test
        if max_val_size > 0 and len(val_dataset) > max_val_size:
            val_dataset = torch.utils.data.Subset(val_dataset, range(max_val_size))
        print(f"Nombre d'échantillons dans le dataset d'entraînement : {len(train_dataset)}")
        print(f"Nombre d'échantillons dans le dataset de validation : {len(val_dataset)}")


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fct = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0

            cpt = 1
            for batch in train_loader:
                print(f"Batch {cpt}/{len(train_loader)}")
                input_ids, attention_mask, *labels = [b.to(self.device) for b in batch]

                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                losses = []
                for i, aspect in enumerate(["Prix", "Cuisine", "Service", "Ambiance"]):
                    loss = loss_fct(logits[aspect], labels[i])
                    losses.append(loss)
                total_loss_batch = sum(losses)
                
                total_loss_batch.backward()
                optimizer.step()

                total_loss += total_loss_batch.item()

                cpt += 1

            print(f"Loss: {total_loss / len(train_loader)}")
            self.validation(val_loader)

    def validation(self, val_loader):
        """
        Valide le modèle sur les données de validation.
        :param val_loader: DataLoader contenant les données de validation.
        """
        self.model.eval()
        total_correct = {aspect: 0 for aspect in ["Prix", "Cuisine", "Service", "Ambiance"]}
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, *labels = [b.to(self.device) for b in batch]
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                for i, aspect in enumerate(["Prix", "Cuisine", "Service", "Ambiance"]):
                    preds = torch.argmax(logits[aspect], dim=-1)
                    total_correct[aspect] += (preds == labels[i]).sum().item()

                total_samples += labels[0].size(0)

        for aspect, correct in total_correct.items():
            accuracy = correct / total_samples
            print(f"Validation Accuracy for {aspect}: {accuracy:.4f}")
            
    def predict(self, text: str) -> dict[str,str]:
        """
        Prédit les opinions pour chaque aspect d'un avis donné.
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 4 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        encodings = self.tokenizer(
            text, truncation=True, padding=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"])

        preds = {aspect: torch.argmax(logits[aspect], dim=-1).item() for aspect in logits}
        result = {aspect: self.inverse_map_label(pred) for aspect, pred in preds.items()}
        
        return result


    def parse_json_response(self, response: str) -> dict[str, str] | None:
        m = re.findall(r"\{[^\{\}]+\}", response, re.DOTALL)
        if m:
            try:
                jresp = json.loads(m[0])
                for aspect, opinion in jresp.items():
                    if "non exprim" in opinion.lower():
                        jresp[aspect] = "NE"
                return jresp
            except:
                return None
        else:
            return None
























