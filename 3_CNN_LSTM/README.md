# PA 3: Image Captioning


## Contributors
Xuzhe Zhi
Yunhao Li
Zhaochen Zhu

## Task
Experiment with different model and different hyperparameters to see the models' performance. 


## How to run
1. Create your own config file - "YOUR_CONFIG_FILE.json". The template config file is given as "task-1-default-config.json". 
2. In experiment.py line 22, change ROOT_STATS_DIR to './YOUR_DESIRED_LOCATION', this is where you will stores the performance in this experiment
3. In main.py line 14, change exp_name to "YOUR_CONFIG_FILE". 
4. After the preperation Run command: 
```bash
python main.py
```
And a model will be trained and tested, the training and testing logs will be saved to the given directory.

### How you may change the config file:
1. Change "model" - "model_type" to specify which model you want to experiment with. "ResNet" (or everything other than "Custom") will generate model ResNet50-LSTM. "Custom" will generate CNN-LSTM. "DeepCustom" with generate a modified CNN-LSTM with more layers.
2. Experiment with different learning rate. "experiment" - "learning_rate"
3. Caption generation method. Change "generation" - "deterministic" to true to use deterministic method (not recommended). Keep it to false (default) will use multinomial
4. Change temperature when using multinomial. High temperature behaves like uniform distribution while low temperature behaves like deterministic.


## Usage

* Define the configuration for your experiment. See `task-1-default-config.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir.
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training or evaluate performance.

## Files
- `main.py`: Main driver class
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- `dataset_factory.py`: Factory to build datasets based on config
- `model_factory.py`: Factory to build models based on config
- `file_utils.py`: utility functions for handling files
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace
