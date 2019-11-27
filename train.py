from loguru import logger
from pipelines import UganTrainPipeline

if __name__ == "__main__":
    logger.add(sys.stderr, format="{time} {message}", filter="my_module", level="INFO")
    logger.add("logs/train_{time}.log")
    logger.info("Run UGAN train pipeline...")
    try:
        UganTrainPipeline().run(
            training_dataset_A_path="train_dataset/trainA",
            training_dataset_B_path="train_dataset/trainB",
            epochs_num=100,
            batch_size=64,
            num_critic=5,
            learning_rate=1e-4,
            torch_manual_seed=0,
            epoch_save_period=5,
            save_discriminator_net=True,
            model_save_path_prefix="traced_models/trace",
        )
    except:
        logger.exception("Failure error!")
