# Punktewolken_Dynamikerkennung

## Name
Qualitätsprüfung von Gussprodukten mit Hilfe von Bilddaten 

## Description
Mit Hilfe des Prinzips Transfer-Learning wurden zwei CNN-Modelle auf Basis von ResNet18 und ResNet50 für die Qualitätsprüfung von Gussteilen entwickelt und evaluiert. 

## Requirements 
Packages: `torch`, `torchvision`, `numpy`, `matplotlib`, `pandas`, `pathlib`, `sklearn`, `tqdm` 

## Preparation 
Die [Daten](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/data ) wurden in Training-, Validierungs- und Testdaten im Verhältnis 6:2:2 aufgeteilt und in [data](./data) gespeichert. 
Datenbeispiele und die Verteilung von Klassen können mit Hilfe von [visualization.ipynb](./visualization.ipynb) dargestellt werden. 


## Usage
ResNet18 wurde mit Hilfe von [train_ResNet18.ipynb](./train_ResNet18.ipynb) und [test_ResNet18.ipynb](test_ResNet18.ipynb) trainiert und evaluiert, und ResNet50 wurde mit Hilfe von [train_ResNet50.ipynb](./train_ResNet50.ipynb) und [test_ResNet50.ipynb](./test_ResNet50.ipynb) trainiert und evaluiert. 

Das Modell mit der besten Leistung während des Trainingsprozesses wurde in [models](./models) gespeichert. Der in [logs](./logs) gespeicherte Trainingsprozess kann mit Hilfe von [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html ) visualisiert werden. 

[demo.ipynb](./demo.ipynb) zeigt, wie Bilder in [demo](./demo) mit Hilfe eines trainierten Modells klassifiziert werden können. 

Die Ergebnisse der Evaluierung der beiden Modelle können mit Hilfe von [plot.ipynb](./plot.ipynb) präsentiert. 

## License
Dieses Projekt ist unter der MIT-Lizenz lizenziert – Einzelheiten finden Sie in der Datei [LICENSE](./LICENSE). 
