import os
import sys
from src.logger import *
from src.exception import *
from src.constants import *
from src.config.configuration import *
from src.components.model_evaluation_mlflow import *


stage_name="EVALUATION STAGE"

class EvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        try:

            config=ConfigurationManager(CONFIG_FILE_PATH,PARAM_FILE_PATH)
            eval_config=config.get_evaluation_config()
            evaluation1=Evaluation(eval_config)
            evaluation1.evaluation()
            evaluation1.save_score()
            #evaluation1.log_into_mlflow()
            #not needed again and again
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/shrianshsingh20/Kidney_tumor_Detection.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"]="shrianshsingh20"
    os.environ["MLFLOW_TRACKING_PASSWORD"]="1ed862e5af9a1dde281263c79e0305550ac28646"
    try:
        logging.info(f"stage {stage_name} started ")
        obj = EvaluationPipeline()
        obj.main()
        logging.info(f"stage {stage_name} completed")
    except Exception as e:
        logging.info("error occured in runnig of the model evaluation pipeline")
        raise CustomException(e,sys)
    
