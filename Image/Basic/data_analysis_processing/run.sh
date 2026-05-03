#! /bin/bash
# get vis:
    # duplicate.html/components.html/outlier.html
    # cluster.png/broken images/duplicate images/outliers images
python main.py 0 /mnt/workspace/hjliu/apc_chile/DMS /mnt/workspace/hjliu/apc_chile/results -d Fasle -n 20

# get delete duplicate image in root path,be careful when calling this function 
# python main.py 1 /mnt/workspace/hjliu/apc_chile/DMS /mnt/workspace/hjliu/apc_chile/results  

# retrieval image 
# python main.py 2 /mnt/workspace/hjliu/apc_chile/DMS /mnt/workspace/hjliu/apc_chile/results -1 /mnt/workspace/hjliu/test/query.jpg -k 10
