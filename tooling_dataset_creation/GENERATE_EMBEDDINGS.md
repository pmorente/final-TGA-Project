# 1. all-MiniLM-L6-v2 (384 dimensions) - Fastest, smallest
python generateEmbeddings.py input.csv -m all-MiniLM-L6-v2 -o embeddings.bin

# 2. all-mpnet-base-v2 (768 dimensions) - Better quality, medium size
python generateEmbeddings.py input.csv -m all-mpnet-base-v2 -o embeddings.bin

# 3. all-roberta-large-v1 (1024 dimensions) - Best quality, largest
python generateEmbeddings.py input.csv -m all-roberta-large-v1 -o embeddings.bin


Faster with GPU:

python generateEmbeddings.py input.csv -m all-MiniLM-L6-v2 --mode batch --batch-size 64 -o embeddings.bin
python generateEmbeddings.py input.csv -m all-mpnet-base-v2 --mode batch --batch-size 32 -o embeddings.bin
python generateEmbeddings.py input.csv -m all-roberta-large-v1 --mode batch --batch-size 16 -o embeddings.bin

