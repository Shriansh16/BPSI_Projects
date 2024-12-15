import os
import sys
import gdown
import zipfile
from src.logger import *
from src.exception import *
from src.entity.config_entity import *


class DataIngestion:
    def __init__(self,config):
        self.config=config
    def download_file(self):
        try:
            dataset_url=self.config.source_url
            zip_file_path=self.config.local_data_file
            os.makedirs('artifacts/data_ingestion',exist_ok=True)
            logging.info("downloading dataset from {url_path} into file {zip_file_path}")
            file_id=dataset_url.split('/')[-2]
            prefix_url='https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix_url+file_id,zip_file_path)
            logging.info("dataset downloaded successfully")
        except Exception as e:
            logging.info("ERROR OCCURED IN DOWNLOADING THE DATASET")
            raise CustomException(e,sys)
    def unzip_file(self):
        try:
            unzip_path=self.config.unzip_dir
            os.makedirs(unzip_path,exist_ok=True)
            logging.info("unzipping the downloaded file")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logging.info("unzipping of file successfull")
        except Exception as e:
            logging.info("ERROR OCCURED IN UNZIPPING THE FILE")
            raise CustomException(e,sys)
            
