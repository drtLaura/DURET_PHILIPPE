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

### **Hyper-paramètres**  
- **Taille du Batch** : 32 (max)  
- **Nombre d’Époques** : 10 à 18 (selon les expérimentations)  
- **Learning Rate** : `5e-5`  
- **Dropout** : [Dropout rate] à préciser  

---

### **Ressources Utilisées**  
- **Infrastructure Matérielle** :  
  - GPU compatible (CUDA 12.2)  
- **Framework et Bibliothèques** :  
  - PyTorch  
  - Hugging Face Transformers  
  - PyTorch Lightning  

---

## **Meilleur résultat**  
- **Nombre d’Époques** :  
- **Taille du Batch** :  
- **Échantillons** :  
- **Résultats** :  
  - **Précisions pour toutes les itérations** : 
  - **Précision Moyenne Macro (AVG MACRO ACC)** : **X%**  
  - **Temps Total d’Exécution** : **Xs**  


Les expérimentations ont été réalisées sur différentes configurations (nombre d’échantillons, époques). Voici les résultats obtenus :  

### **1ère Configuration**  
- **Nombre d’Époques** : 10  
- **Taille du Batch** : 32  
- **Échantillons** : 100 et 400  
- **Résultats** :  
  - **Précisions pour toutes les itérations** : [75.25, 75.5, 78.5]  
  - **Précision Moyenne Macro (AVG MACRO ACC)** : **76.42%**  
  - **Temps Total d’Exécution** : **868.9s**  

---

### **2ᵉ Configuration**  
- **Nombre d’Époques** : 15  
- **Taille du Batch** : 32 (max)  
- **Échantillons** : 300 et 1000  
- **Résultats** :  
  - **Précisions pour toutes les itérations** : [80.17, 79.83, 80.08]  
  - **Précision Moyenne Macro (AVG MACRO ACC)** : **80.03%**  
  - **Temps Total d’Exécution** : **3238.2s**  

---

### **3ᵉ Configuration**  
- **Nombre d’Époques** : 18  
- **Taille du Batch** : 32  
- **Échantillons** : 150 et 500  
- **Résultats** :  
  - **Précisions pour toutes les itérations** : [78.0, 80.17, 78.0]  
  - **Précision Moyenne Macro (AVG MACRO ACC)** : **78.72%**  
  - **Temps Total d’Exécution** : **1807.7s**  

---

### **Résumé Global**  

--- 
