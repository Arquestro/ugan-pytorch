from loguru import logger
from pipelines import UganDatasetInferencePipeline

if __name__ == "__main__":
    logger.add(sys.stderr, format="{time} {message}", filter="my_module", level="INFO")
    logger.add("logs/inference_{time}.log")
    logger.info("Run UGAN inference pipeline...")
    try:
        UganDatasetInferencePipeline().run(
            target_dataset_path="train_dataset/trainA/",
            trace_filepath="trace_generator_epoch95.pt",
            output_directory_path="output/",
            batch_size=64,
            torch_manual_seed=0,
            use_cuda=True,
        )
    except:
        logger.exception("Failure error!")
