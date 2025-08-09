import pandas as pd
from langchain_core.documents import Document

# To convert Csv File to Document and it will have only 2 fields review and title
#bcz we are concerend only with these 2 fields
#but here we have just defined function, it is called from data_ingestion.py
class DataConverter:
    def __init__(self,file_path:str):
        self.file_path = file_path

    def convert(self):
        df = pd.read_csv(self.file_path)[["product_title","review"]]   

        docs = [
            Document(page_content=row['review'] , metadata = {"product_name" : row["product_title"]})
            for _, row in df.iterrows()
        ]

        return docs
