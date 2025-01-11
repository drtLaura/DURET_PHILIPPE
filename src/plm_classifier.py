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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # tokenizer utilisé pour convertir les textes en séquences numériques compréhensibles par le modèle
        self.model = MultiAspectClassifier(model_name).to(self.device) # modèle utilisé pour prédire les opinions
        self.n_test = n_test
        self.n_train = n_train
        print(f"n_test : {n_test}")
        print(f"n_train : {n_train}")
        

    def prepare_data(self, data: list[dict], max_length: int = 512, is_train: bool = True):
        texts = [item["Avis"] for item in data] # liste des avis
        labels = {
            aspect: [self.map_label(item[aspect]) for item in data]
            for aspect in ["Prix", "Cuisine", "Service", "Ambiance"]
        } # dictionnaire des étiquettes pour chaque aspect
        
        if self.n_train > 0 and is_train: # si on a un nombre d'échantillons d'entraînement spécifié et qu'on est en phase d'entraînement
            texts = texts[:self.n_train] # on ne garde que les n_train premiers avis
            for aspect in labels: # on ne garde que les n_train premières étiquettes pour chaque aspect
                labels[aspect] = labels[aspect][:self.n_train]
                              
        encodings = self.tokenizer( # on convertit les textes en séquences numériques pour les passer au modèle
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ) 
        labels = {aspect: torch.tensor(values) for aspect, values in labels.items()} # on convertit les étiquettes en tenseurs torch
       
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"], encodings["attention_mask"],
            *[labels[aspect] for aspect in labels]
        )  # on crée un dataset torch contenant les séquences numériques, les degre d'importance et les étiquettes pour chaque aspect
        return dataset

    @staticmethod
    def map_label(label: str) -> int:   # mappe une étiquette texte vers un index numérique
        """
        Mappe une étiquette texte vers un index numérique.
        """
        label_map = {"Positive": 0, "Négative": 1, "Neutre": 2, "NE": 3}
        return label_map[label]

    @staticmethod 
    def inverse_map_label(index: int) -> str: # inverse de map_label : mappe un index numérique vers une étiquette texte
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
        # Préparation des données d'entraînement (train et val)
        train_dataset = self.prepare_data(data=train_data, is_train=True)
        val_dataset = self.prepare_data(data=val_data, is_train=False)
        max_val_size = self.n_test # nombre d'échantillons de validation (meme que pour n_test ?)
        if max_val_size > 0 and len(val_dataset) > max_val_size:
            val_dataset = torch.utils.data.Subset(val_dataset, range(max_val_size))
        print(f"Nombre d'échantillons dans le dataset d'entraînement : {len(train_dataset)}")
        print(f"Nombre d'échantillons dans le dataset de validation : {len(val_dataset)}")

        # DataLoaders pour charger les données en lots
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr) # pour mettre à jour le poids
        loss_fct = nn.CrossEntropyLoss() # fonction de perte = comparer les prédictions aux labels

        for epoch in range(epochs): # boucle sur les époques (nombre de fois où on parcourt tout le dataset train)
            print(f"Epoch {epoch + 1}/{epochs}") 
            self.model.train() # on met le modèle en mode entraînement
            total_loss = 0

            cpt = 1 # compteur pour afficher le numéro du batch
            for batch in train_loader: # pour chaque batch
                print(f"Batch {cpt}/{len(train_loader)}") 
                input_ids, attention_mask, *labels = [b.to(self.device) for b in batch] # on déplace les données sur le device

                optimizer.zero_grad() # on remet à zéro les gradients
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask) # on fait une prédiction

                losses = [] # liste pour stocker les pertes pour chaque aspect
                for i, aspect in enumerate(["Prix", "Cuisine", "Service", "Ambiance"]):
                    loss = loss_fct(logits[aspect], labels[i])
                    losses.append(loss)
                total_loss_batch = sum(losses) # somme des pertes pour chaque aspect
                
                total_loss_batch.backward() # rétropropagation
                optimizer.step() # mise à jour des poids

                total_loss += total_loss_batch.item() # on ajoute la perte du batch à la perte totale

                cpt += 1 # on incrémente le compteur de batch

            print(f"Loss: {total_loss / len(train_loader)}") # on affiche la perte moyenne pour l'époque
            self.validation(val_loader) # on valide le modèle sur les données de validation

    def validation(self, val_loader):
        """
        Valide le modèle sur les données de validation.
        :param val_loader: DataLoader contenant les données de validation.
        """
        self.model.eval() # on met le modèle en mode évaluation
        total_correct = {aspect: 0 for aspect in ["Prix", "Cuisine", "Service", "Ambiance"]} # nombre total de prédictions correctes
        total_samples = 0 # nombre total d'échantillons

        with torch.no_grad():  # pas de calcul de gradient pour la validation
            for batch in val_loader:  # pour chaque batch
                input_ids, attention_mask, *labels = [b.to(self.device) for b in batch] # on déplace les données sur le device
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask) # on fait une prédiction

                for i, aspect in enumerate(["Prix", "Cuisine", "Service", "Ambiance"]): # pour chaque aspect
                    preds = torch.argmax(logits[aspect], dim=-1) # on récupère les prédictions
                    total_correct[aspect] += (preds == labels[i]).sum().item()  # on met à jour le nombre de prédictions correctes

                total_samples += labels[0].size(0) # on met à jour le nombre total d'échantillons

        for aspect, correct in total_correct.items(): # pour chaque aspect
            accuracy = correct / total_samples # on calcule la précision
            print(f"Validation Accuracy for {aspect}: {accuracy:.4f}") # on affiche la précision
            
    def predict(self, text: str) -> dict[str,str]:
        """
        Prédit les opinions pour chaque aspect d'un avis donné.
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 4 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        encodings = self.tokenizer( # on convertit le texte en séquence numérique
            text, truncation=True, padding=True, max_length=512, return_tensors="pt"
        ).to(self.device) # on déplace les données sur le device

        with torch.no_grad(): # pas de calcul de gradient
            logits = self.model(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"]) # on fait une prédiction

        preds = {aspect: torch.argmax(logits[aspect], dim=-1).item() for aspect in logits} # on récupère les prédictions
        result = {aspect: self.inverse_map_label(pred) for aspect, pred in preds.items()} # on mappe les prédictions en texte
        
        return result # on retourne le résultat


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
























