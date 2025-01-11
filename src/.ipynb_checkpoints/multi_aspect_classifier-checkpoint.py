from transformers import AutoModel
import torch.nn as nn

class MultiAspectClassifier(nn.Module):
    # Architecture multi task pour gérer chaque aspect en simultané 
    
    def __init__(self, model_name: str, num_labels: int = 4):
        super(MultiAspectClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.aspect_heads = nn.ModuleDict({
            "Prix": nn.Linear(self.backbone.config.hidden_size, num_labels),
            "Cuisine": nn.Linear(self.backbone.config.hidden_size, num_labels),
            "Service": nn.Linear(self.backbone.config.hidden_size, num_labels),
            "Ambiance": nn.Linear(self.backbone.config.hidden_size, num_labels),
        })

    def forward(self, input_ids, attention_mask):
        backbone_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = backbone_output.last_hidden_state[:, 0, :]  # CLS token
        logits = {aspect: head(hidden_state) for aspect, head in self.aspect_heads.items()}
        return logits
