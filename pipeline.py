import kfp
from kfp.dsl import pipeline
from components.preprocess import preprocess
from components.train import train
from components.evaluate import evaluate

@pipeline
def iris_pipeline():
    preprocess_task = preprocess(output_path="/mnt/data")
    
    # Use the output path returned by preprocess
    train_task = train(input_path=preprocess_task.output, model_path="/mnt/data")
    
    # Reference the model file output from the train task
    evaluate(input_path=preprocess_task.output, model_path=train_task.output, metrics_path="/mnt/data")

# Compile the pipeline
kfp.compiler.Compiler().compile(iris_pipeline, "iris_pipeline.yaml")
