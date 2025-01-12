# **Projet - Fouille de Texte**  

## **Auteur(s)**  
- **Laura DURET**  
- **Aurore PHILIPPE**  

---

## **Description du Classifieur**  

Le classifieur repose sur une approche avancée de fine-tuning de modèles de langage pré-entraînés (**PLMFT - Pretrained Language Model Fine-Tuning**).  
Nous avons exploité **CamemBERT-base**, un modèle spécifiquement conçu pour la langue française, comme fondation.  
Ce modèle est capable de classer les opinions exprimées sur quatre aspects distincts : **Prix**, **Cuisine**, **Service**, et **Ambiance**, grâce à une architecture multitâche optimisée.  

---

### **Type de Représentation**  
Le modèle utilise les **représentations contextualisées** générées par le backbone **CamemBERT-base**.  
Ces représentations sont obtenues via le token spécial `[CLS]`, qui capture l’information contextuelle globale pour chaque séquence d’entrée.  
Cette stratégie garantit une compréhension fine des nuances linguistiques propres à la langue française, en tenant compte du contexte spécifique de chaque mot.  

---

### **Architecture du Modèle**  

#### **1. Backbone Principal**  
- **CamemBERT-base** est utilisé pour produire des vecteurs de caractéristiques riches et contextualisés pour chaque séquence d’entrée.  

#### **2. Têtes de Classification**  
- Quatre **couches linéaires indépendantes** sont mises en œuvre, une par aspect (**Prix**, **Cuisine**, **Service**, **Ambiance**).  
- Chaque tête de classification génère des scores (**logits**) pour quatre classes possibles :  
  - **Positive**, **Négative**, **Neutre**, et **NE** (Non Exprimé).  

#### **3. Flexibilité Multitâche**  
- Le modèle traite simultanément les différents aspects d’un même texte, offrant une analyse complète et efficace des opinions.  

---

## **Hyperparamètres meilleur résultat**  
- **Nombre d’Époques** :  15
- **Taille du Batch** :  32
- **Échantillons** :  Toutes les données disponibles dans train, test et val
- **Résultats** :  
  - **Précisions pour toutes les itérations** : 
  - **Précision Moyenne Macro (AVG MACRO ACC)** : **X%**  
  - **Temps Total d’Exécution** : **Xs**  

---

### **Ressources Utilisées**  
- **Infrastructure Matérielle** :  
  - GPU compatible (CUDA 12.2)  
- **Framework et Bibliothèques** :  
  - PyTorch  
  - Hugging Face Transformers  
  - PyTorch Lightning  

---


## **Autre résultat**  

- **Modèle** :  gemma2:2b
- **Résultats** :  
ALL RUNS ACC: [67.07]
AVG MACRO ACC: 67.07
TOTAL EXEC TIME: 1419.5

- **Modèle** :  smollm2:1.7b
- **Résultats** :  
ALL RUNS ACC: [37.72]
AVG MACRO ACC: 37.72
TOTAL EXEC TIME: 1096.7
  
