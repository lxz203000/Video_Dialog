# video_dialog

The implementation of paper Dual Temporal Grounding-enhanced Video Dialog

## Data Processing
Please download the required data from the [homepage](https://github.com/dialogtekgeek/AVSD-DSTC10_Official/tree/main) of DSTC10 and place it in `data/`, including:
* test_set4DSTC7-AVSD.json
* test_set4DSTC8-AVSD.json
* train_set4DSTC8-AVSD+reason
* valid_set4DSTC8-AVSD+reason

If you want to additionally use audio features:
* vggish.tgz
* vggish_testset.tgz

As for video features, please download the RGB frames from [Charades](https://prior.allenai.org/projects/charades) and use the pre-trained S3D model to extract features.

## Requirements
* python==3.6.9
* torch==1.7.0+cu92
* tqdm
* boto3
* requests
* pandas
* nlg-eval (Install Java 1.8.0 (or higher) firstly)

```
conda create -n DTGVD python=3.6.9 tqdm boto3 requests pandas
conda activate DTGVD
pip install torch==1.7.1+cu92
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```

## Pipeline
Our model consists of two parts, i.e. *grounding* and *dialog*, which can be switched directly by changing `--task_type`.
To complete the whole process, please first use the *grounding* part to get the timestamps corresponding to each QA and save it as a pickle file. 
In the evaluation part of *dialog*, use `--grounding_results` to point to the pickle file to get the final results.

## Training
The hyperparameters are displayed in **main.py**, if you want to use the default hyperparameters, you can run directly:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 10000 main.py
```

## Evaluation
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 10000 main.py
```







