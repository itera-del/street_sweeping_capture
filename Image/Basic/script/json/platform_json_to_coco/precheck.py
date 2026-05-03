
import os

def get_data_id_list(datadir):
    imgdir = '{}/images'.format(datadir)
    tagdir = '{}/json_v0'.format(datadir)
    
    imglist = os.listdir(imgdir)
    taglist = os.listdir(tagdir)

    imglist = [i for i in imglist if i.endswith('jpg')]
    taglist = [i for i in taglist if i.endswith('json')]

    img_id_list = [i[:-4] for i in imglist]
    tag_id_list = [i[:-5] for i in taglist]
    
    data_id_list = list(set(img_id_list) & set(tag_id_list))
    return data_id_list
    
if __name__ == '__main__':
    
    date = [
        '20220425', 
        '20220429', 
        '20220512', 
        'ADASDetSeg', 
        'BSDsidewaysSegOld', 
        
    ]
    
    count = 0
    for d in date:
        datadir = '/lirui/DATA/BSDsideways/{}'.format(d)
        data_id_list = get_data_id_list(datadir)
        count += len(data_id_list)
        print(d, len(data_id_list))
    print('all: ', count)

    
    
