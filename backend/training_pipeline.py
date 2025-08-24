import queue
import threading
from backend.data import load_and_preprocess_data
from backend.model_setup import setup_model_and_tokenizer
from backend.trainer import Trainer

class TrainingPipeline:
    """
    A class to encapsulate the entire training pipeline, from data loading to model training.
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.progress_queue = queue.Queue()

    def _setup(self):
        """
        Sets up the model, tokenizer, and datasets.
        """
        self.model, self.tokenizer = setup_model_and_tokenizer(self.config)
        self.train_dataset, self.eval_dataset = load_and_preprocess_data(self.config, self.tokenizer)

    def _train(self):
        """
        Initializes and runs the training process.
        """
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            config=self.config,
            progress_queue=self.progress_queue
        )
        self.trainer.train()

    def run_in_thread(self):
        """
        Runs the training pipeline in a separate thread to keep the UI responsive.
        """
        def target():
            self._setup()
            self._train()

        thread = threading.Thread(target=target)
        thread.start()
        return thread
