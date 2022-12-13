python scripts/train.py --dataset=twitter --algo=bert --max-epoch=100 > twitter_train_bert.txt 
python scripts/train.py --dataset=twitter --algo=bert-large  --max-epoch=100 > twitter_train_bert-large.txt
python scripts/train.py --dataset=twitter --algo=distilbert --max-epoch=100 > twitter_train_distilbert.txt 
python scripts/test.py --dataset=twitter --algo=bert  > twitter_test_bert.txt
python scripts/test.py --dataset=twitter --algo=bert-large  > twitter_test_bert-large.txt
python scripts/test.py --dataset=twitter --algo=distilbert  > twitter_test_distilbert.txt

python scripts/train.py --dataset=wikipedia --algo=bert --max-epoch=100 > wikipedia_train_bert.txt
python scripts/train.py --dataset=wikipedia --algo=bert-large --max-epoch=100 > wikipedia_train_bert-large.txt
python scripts/train.py --dataset=wikipedia --algo=distilbert --max-epoch=100 > wikipedia_train_distilbert.txt
python scripts/test.py --dataset=wikipedia --algo=bert  > wikipedia_test_bert.txt
python scripts/test.py --dataset=wikipedia --algo=bert-large  > wikipedia_test_bert-large.txt
python scripts/test.py --dataset=wikipedia --algo=distilbert  > wikipedia_test_distilbert.txt