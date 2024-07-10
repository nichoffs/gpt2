git clone https://github.com/nichoffs/gpt2.git
cd gpt2
pip install -r requirements.txt
pip install -e .
python3 src/data/download_tinystories.py ../tinystories/datasets/tinystories
python3 src/data/train.py -p checkpoints/lambda_run -s 21 -d ../tinystories/datasets/tinystories
