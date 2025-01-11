from pandas import DataFrame
from tqdm import tqdm

from config import Config

from llm_classifier import LLMClassifier
from plm_classifier import PLMClassifier

class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    # METHOD: str = 'LLM'  
    METHOD: str = 'PLMFT' # 'PLMFT' (for Pretrained Language Model Fine-Tuning)

    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        self.cfg = cfg

        #############################################################################################
        # INITIALISATION LLM
        #############################################################################################
        # self.classifier = LLMClassifier(cfg)

        #############################################################################################
        # INITIALISATION PLMFT
        #############################################################################################
        # print("Init PLMFT...")
        self.classifier = PLMClassifier(n_test=cfg.n_test, n_train=cfg.n_train, model_name="camembert-base", device=cfg.device)



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        :param train_data:
        :param val_data:
        :param device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu
        :return:
        """
        #############################################################################################
        # ENTRAÎNEMENT POUR LLM
        #############################################################################################
        # Mettre tout ce qui est nécessaire pour entrainer le modèle ici, sauf si methode=LLM en zéro-shot
        # auquel cas pas d'entrainement du tout
        # pass

        #############################################################################################
        # ENTRAÎNEMENT POUR PLMFT
        #############################################################################################
        # Appel au train (changer les hyperparamètres batch_size, learning_rate, et epochs pour optimiser les performances)
        self.classifier.train(train_data=train_data, val_data=val_data, batch_size=1, epochs=1, lr=5e-5)



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        all_opinions = [] 

        #############################################################################################
        # PREDICTION POUR LLM 
        #############################################################################################
        # Comme ici on utilise un llm avce ollama, on procèdera en traitant les textes d'avis un à un
        # mais si on utilise un PLMFT, il vaut mieux traiter les avis par batch pour que ce soit plus
        # rapide
        # for text in tqdm(texts):
        #     opinions = self.classifier.predict(text)
        #     all_opinions.append(opinions)
        # return all_opinions
    
        #############################################################################################
        # PREDICTION POUR PLMFT
        #############################################################################################
        batch_size = 16  # Taille des lots à ajuster 
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_opinions = self.classifier.predict(batch_texts)  # Prédiction par lot
            all_opinions.extend(batch_opinions)
            
        return all_opinions

