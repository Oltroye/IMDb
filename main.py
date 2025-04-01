<<<<<<< HEAD
import os
from src.analyse import analyse_data
from src.model import train_model
import pandas as pd

def main():
    try:
       
        data_path = os.path.join("data", "IMDB_Dataset.csv")
        
    
        df = pd.read_csv(data_path)
        
        
        analyse_data(df)
        
        train_model(df)
        
    except FileNotFoundError:
        print(f"Erreur: Le fichier {data_path} n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main()
=======
from src.preprocessor import Preprocessor
from src.model import Model

if __name__ == "__main__":
    # Étape 1 : Prétraitement des données
    preprocessor = Preprocessor()
    df = preprocessor.fit_transform("data/IMDB Dataset.csv")

    # Étape 2 : Entraînement et évaluation du modèle
    model = Model()
    model.train(df)
    model.evaluate()
>>>>>>> cleo
