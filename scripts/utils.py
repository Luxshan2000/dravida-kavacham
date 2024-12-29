import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    texts = data['Text'].tolist()
    
    if 'Class' in data.columns:
        labels = data['Class'].apply(lambda x: 1 if x == 'Abusive' else 0).tolist()
    else:
        labels = None
        
    return texts, labels
