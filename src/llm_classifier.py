from jinja2 import Template
from ollama import Client
import re
import json


from config import Config


_PROMPT_TEMPLATE = """Considérez l'avis suivant:

"{{text}}"

Quelle est la valeur de l'opinion exprimée sur chacun des aspects suivants :
- Prix : concerne les prix des plats et boissons 
- Cuisine : concerne la qualité et la quantité de la nourriture et des boissons proposées. 
- Service : concerne la qualité et l'efficacité du service et de l'accueil.
- Ambiance : concerne le style, la qualité de la décoration, le confort, et l'ambiance à l'intérieur du restaurant.

La valeur d'une opinion doit être une des valeurs suivantes: 
- "Positive" : lorsque l'avis contient une ou plusieurs opinions positives sur l'aspect en question, et aucune opinion négative sur cet aspect.
- "Négative" : lorsque l'avis contient seulement une ou plusieurs opinions négatives sur l'aspect en question et aucune opinion positive sur cet aspect.
- "Neutre" : lorsque l'avis exprime au moins une opinion positive et une autre négative sur l'aspect en question.
- "NE" (Non Exprimée) : lorsque l'avis ne contient aucune opinion exprimée sur l'aspect en question.

Voici un exemple :

Avis : "Les prix sont raisonnables, mais la nourriture manque de saveur."
Réponse :
{
    "Prix": "Positive",
    "Cuisine": "Négative",
    "Service": "NE",
    "Ambiance": "NE"
}

La réponse doit se limiter au format json suivant:
{ "Prix": opinion, "Cuisine": opinion, "Service": opinion, "Ambiance": opinion}."""



class LLMClassifier:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Instantiate an ollama client
        self.llmclient = Client(host=cfg.ollama_url)
        # self.model_name = 'llama3.1:latest'
        # self.model_name = 'gemma2:latest'
        #self.model_name = 'gemma2:2b'
        #self.model_name = 'llama3.2:1b'
        self.model_name = 'smollm2:1.7b'
        self.model_options = {
            'num_predict': 500,  # max number of tokens to predict
            'temperature': 0.1,
            'top_p': 0.9,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)


    def predict(self, text: str) -> dict[str,str]:
        """
        Lance au LLM une requête contenant le texte de l'avis et les instructions pour extraire
        les opinions sur les aspects sous forme d'objet json
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 4 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        prompt = self.jtemplate.render(text=text)
        result = self.llmclient.generate(model=self.model_name, prompt=prompt, options=self.model_options)
        response = result['response']
        jresp = self.parse_json_response(response)
        return jresp

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



























