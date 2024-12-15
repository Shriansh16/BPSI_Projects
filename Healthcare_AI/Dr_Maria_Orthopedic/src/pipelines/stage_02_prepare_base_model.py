import sys
from src.config.configuration import *
from src.components.prepare_base_model import *
from src.logger import *
from src.exception import *

STAGE_NAME='PREPARE BASE MODEL'

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            configuration_manager=ConfigurationManager(CONFIG_FILE_PATH,PARAM_FILE_PATH)
            base_model_config=configuration_manager.get_prepare_base_model_config()
            prepare_base_model=PrepareBaseModel(base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    try:
        obj2=PrepareBaseModelPipeline()
        logging.info(f'{STAGE_NAME} HAS STARTED')
        obj2.main()
        logging.info(f'{STAGE_NAME} HAS completed')
    except Exception as e:
        logging.info("error occured in prepare base model pipeline")
        raise CustomException(e,sys)
    
    

        