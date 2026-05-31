# Rapport de Projet

## Introduction

Présenter le problème de classification d'objets 3D à partir de nuages de points et les objectifs de l'étude.

## Jeu de données et prétraitement

Décrire ShapeNet via Cap3D, le format `.ply`, les coordonnées XYZ, les couleurs RGB, la stratégie de sous-échantillonnage et les divisions train/validation/test.

## Architectures choisies

Présenter PointGCN et RS-CNN, ainsi que les choix d'entrée `[B, N, C]` avec `C=6` pour XYZRGB.

## Baseline centralisée

Décrire l'entraînement centralisé, les hyperparamètres, les métriques et le protocole d'évaluation.

## Apprentissage fédéré horizontal IID/non-IID

Expliquer les partitions clients, l'entraînement local manuel en PyTorch et l'agrégation FedAvg.

## Apprentissage fédéré vertical

Décrire la séparation XYZ/RGB, les embeddings envoyés au serveur et les garanties de non-échange des données brutes.

## Distillation de connaissances

Présenter le modèle enseignant centralisé, le modèle étudiant, la perte CE + KL, `alpha` et la température `tau`.

## Résultats expérimentaux

Inclure les tableaux, courbes d'apprentissage, matrices de confusion et métriques principales.

## Analyse comparative

Comparer les configurations centralisée, fédérée IID, fédérée non-IID, VFL et distillation.

## Conclusion

Résumer les résultats, les limites et les perspectives d'amélioration.

