from transformers import AutoModel
import torch.nn as nn

class MultiAspectClassifier(nn.Module):
    # Architecture multi task pour gérer chaque aspect en simultané 
    
    def __init__(self, model_name: str, num_labels: int = 4): # num labels = 4 car on a 4 classes (Positive, Négative, Neutre, NE)
        super(MultiAspectClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name) # chargement du modèle pré-entrainé (ici camembert-base)
        # transforme la représentation générale (sortie du backbone) en une prédiction pour un aspect particulier (notre tache spécifique)
        self.aspect_heads = nn.ModuleDict({ 
            "Prix": nn.Linear(self.backbone.config.hidden_size, num_labels), # prédictions pour chaque aspect = produisent des scores (logits) pour les 4 classes de chaque aspect
            "Cuisine": nn.Linear(self.backbone.config.hidden_size, num_labels),
            "Service": nn.Linear(self.backbone.config.hidden_size, num_labels),
            "Ambiance": nn.Linear(self.backbone.config.hidden_size, num_labels),
        })

    def forward(self, input_ids, attention_mask): # input_ids = séquences tokenisées, attention_mask = indique les tokens à prendre en compte
        backbone_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask) # sortie du backbone
        hidden_state = backbone_output.last_hidden_state[:, 0, :]  # CLS token (représentation de la séquence entière)
        logits = {aspect: head(hidden_state) for aspect, head in self.aspect_heads.items()} # prédiction pour chaque aspect
        return logits # dictionnaire contenant les prédictions pour chaque aspect
