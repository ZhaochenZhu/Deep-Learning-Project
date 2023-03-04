# PA 4: Transformers


## Contributors
Lainey Liu  
Mingkun Sun  
Xuzhe Zhi  
Yunhao Li  
Zhaochen Zhu  

## Task
Experiment with pre-trained BERT model, tune the hyper-parameters to achieve a good performance classification objective; The expected test acccuracy of the tuned baseline model on amazon massive intent dataset is around 84%.

The Custom model improve the model with some fine-tuning techniques and to understand how to use the techniques; 

The SupContrast model use supervised constrastive loss to train BERT.


## How to run
First run the following command to download the package and dataset needed.  
```bash
pip install -r requirements.txt
```
Then you will be able to run the experiment  
### Brief instruction with fine-tuned hyper-parameters: 
1. Baseline model:
```bash
python main.py --n-epochs 20 --do-train --batch-size 32 --learning-rate 0.0001
```
2. Custom Model (with warm up schedular (default) and 3 reinitialize layers):
```bash
python main.py --n-epochs 20 --do-train --task custom --reinit_n_layers 3 --batch-size 32 --learning-rate 0.0001
```
3. Contrastive learning
SupCon Model:
```bash
python main.py --n-epochs 15 --do-train --task supcon --batch-size 64 --learning-rate 0.0001 --temperature 0.07 --drop-rate 0.9
```
4. SimCLR Model (with unsupervised cnotrastive loss and temperature 0.07):
```bash
python main.py --n-epochs 35 --do-train --task supcon --batch-size 100 --learning-rate 0.0001 --supconloss false --temperature 0.07 --drop-rate 0.9
```
### Run you own experiment:
If you want to experiment with your own hyper-parameters, run command: python main.py with your own arguments.  
Detailed instruction about different arguments and how to use them are provided in arguments.py.

Notice running the command line for model training will automatically sava the final model as a .pt file in the same directory as this project. The model will be around 1.3 - 1.5 GB.

To Generate a Plot of the Embedding of baseline model project to 2D space using UMAP: Change task to baselinedraw while keeping all other parameter the same.

To Generate a Plot of embedding of Supcon model project to 2D space using UMAP: CHange task to supcondraw and keep all other parameters the same.
