mkdir assets
mkdir results


#uncomment the below line to install required transformaer package, umap
#pip install -r requirements.txt
#pip install umap-learn

#untuned Baseline
python main.py --n-epochs 8 --do-train

#Fine Tuned Baseline
#Will Store the model, Watch out for memory / storage
python main.py --n-epochs 20 --do-train --batch-size 32 --learning-rate 0.0001
#To Draw the Embedding Of baseline Model
python main.py --n-epochs 20 --do-train --task baselinedraw --batch-size 32 --learning-rate 0.0001

#Fine Tuned Custom Model
#with warm up schedular (default) and 3 reinitialize layers
python main.py --n-epochs 20 --do-train --task custom --reinit_n_layers 3 --batch-size 32 --learning-rate 0.0001


#Fine Tune SupCon Model
python main.py --n-epochs 15 --do-train --task supcon --batch-size 64 --learning-rate 0.0001 --temperature 0.07 --drop-rate 0.9
#Draw Embedding print path to the stored image
python main.py --n-epochs 15 --do-train --task supcondraw --batch-size 64 --learning-rate 0.0001 --temperature 0.07 --drop-rate 0.9

#Fine Tune SimCSE Model
python main.py --n-epochs 35 --do-train --task supcon --batch-size 100 --learning-rate 0.0001 --supconloss false --temperature 0.07 --drop-rate 0.9
#Draw Embedding print path to the stored image
python main.py --n-epochs 35 --do-train --task supcondraw --batch-size 100 --learning-rate 0.0001 --supconloss false --temperature 0.07 --drop-rate 0.9



