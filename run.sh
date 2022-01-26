python -m main.main.py





nohup python -m torch.distributed.launch --nproc_per_node=2 main.main.py >> blog.txt 2>&1 &