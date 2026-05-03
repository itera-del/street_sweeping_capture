from src.model import MODEL
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Run Auto_Dinov2_Processing')
    parser.add_argument('task_id', type=int, help='task style id')
    parser.add_argument('images_path', type=str, help='root path to images')
    parser.add_argument('results_path', type=str, help='root folder to save results')
    parser.add_argument('-q', '--query_image', type=str, help='path to a query image file')
    parser.add_argument('-d', '--is_dinov2', type=bool, default=False, help='whether use dinov2 to get feature vectors')
    parser.add_argument('-n', '--num', type=int, default=20, help='save n image in processing method')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='The number of nearest neighbors to search for image')

    args = parser.parse_args()
    return args


class Run():
    def __init__(self,args):
        '''
        :param imagepath: root path of checking images
        :param results_path:save path of results
        :param id_dinov2: whether use dinov2 to get feature vectors
        :param num: data_processing classmethod to save num images
        '''
        self.images_path = args.images_path
        self.results_path = args.results_path
        self.top_k = args.topk
        self.num = args.num
        self.query_path = args.query_image
        self.model = MODEL(self.images_path, self.results_path, is_dinov2=args.is_dinov2)
        if len(os.listdir(os.path.join(self.results_path, 'work_dir'))) == 0:
            self.model.create_model()
            print('###******create model sucess********###')

    # get duplicate.html、components.html、outlier.html、cluster.png
    # def forward_vis(self):
    #     self.model.vis_processing()

    def forward_processing(self):
        self.model.vis_processing()
        self.model.data_processing(self.num)

    def forward_retrieval(self):
        self.model.retrieval(self.top_k, self.query_path)

    # delete duplicate images in root images path
    def forward_delete_duplicate(self):
        self.model.delete_duplicate_images()

if __name__ == '__main__':
    args = parse_args()
    model = Run(args)
    # model.forward_vis()
    if args.task_id == 0:
        model.forward_processing()
    elif args.task_id == 1:
        model.forward_delete_duplicate()
    else:
        model.forward_retrieval()
