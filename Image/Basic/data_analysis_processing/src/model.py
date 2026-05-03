import fastdup
import os
import shutil
from tqdm import tqdm
import concurrent.futures
from util import *
import glob


class MODEL():
    def __init__(self, path, result, is_dinov2=False):
        '''
        :param path: input image path
        :param result: save result path
        '''
        self.path = path
        self.result = result
        self.raw_images = [os.path.join(path, i) for i in os.listdir(path) if
                           i.endswith(('jpg', 'png', 'jpeg'))]
        if not os.path.exists(os.path.join(self.result, 'work_dir')):
            os.mkdir(os.path.join(self.result, 'work_dir'))
        if not os.path.exists(os.path.join(self.result, 'vis')):
            os.mkdir(os.path.join(self.result, 'vis'))
        self.fd = fastdup.create(input_dir=path, work_dir=os.path.join(self.result, 'work_dir'))
        self.is_dinov2 = is_dinov2

    def create_model(self, cc_threshold=0.8):
        '''
        :param cc_threshold: ccthreshold:Threshold for running connected components to find clusters of similar images. Allowed values 0->1
        '''
        if self.is_dinov2:
            self.fd.run(model_path='dinov2b', cc_threshold=cc_threshold, d=786)
        else:
            self.fd.run(ccthreshold=cc_threshold)

        # Get a summary of the run showing potentially problematic files
        print(self.fd.summary())

    def find_borken_images(self, num=20):
        broken_images = self.fd.invalid_instances()
        list_of_broken_images = broken_images['filename'].to_list()
        if num != -1:
            num = num if len(list_of_broken_images) > num else len(list_of_broken_images)
            list_of_broken_images = list_of_broken_images[:num]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(self.copy_broken_image, list_of_broken_images), total=len(list_of_broken_images)))

    def find_duplicate_images(self, num=20):
        connected_components_df, _ = self.fd.connected_components()
        clusters_df = get_clusters(connected_components_df)
        cluster_images_to_keep = []
        list_of_duplicates = []

        for cluster_file_list in clusters_df.filename:
            # keep first file, discard rest
            keep = cluster_file_list[0]
            discard = cluster_file_list[1:]

            cluster_images_to_keep.append(keep)
            list_of_duplicates.extend(discard)

        print(f"Found {len(set(list_of_duplicates))} highly similar images to discard")
        if num != -1:
            num = num if len(list_of_duplicates) > num else len(list_of_duplicates)
            list_of_duplicates = list_of_duplicates[:num]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(self.copy_duplicate_image, list_of_duplicates), total=len(list_of_duplicates)))

    def find_outlier_image(self, thresh=None, num=20):
        '''
        Outliers are computed using the fastdup tool, by embedding each image to a short feature vector,
        finding top k similar neighbors and finding images that are further away from all other images.
        i.e. 0.05 means top 5% of the images that are further away from the rest of the images are considered outliers.
        :return:
        '''
        outlier_df = self.fd.outliers()
        if thresh:
            list_of_outliers = outlier_df[outlier_df.distance < thresh].filename_outlier.tolist()
        else:
            list_of_outliers = outlier_df.filename_outlier.tolist()
        if num != -1:
            num = num if len(list_of_outliers) > num else len(list_of_outliers)
            list_of_outliers = list_of_outliers[:num]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(self.copy_outliers_image, list_of_outliers), total=len(list_of_outliers)))

    def find_retrieval_images(self, cluster_num, query_image_path):
        if self.is_dinov2:
            fastdup.init_search(cluster_num, os.path.join(self.result, 'work_dir'), d=786, model_path='dinov2b')
        else:
            fastdup.init_search(cluster_num, os.path.join(self.result, 'work_dir'))

        df = fastdup.search(query_image_path)
        retrieval_results = df['to'].to_list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(self.copy_retrieval_image, retrieval_results), total=len(retrieval_results)))

    def remove_duplicate_images(self):
        connected_components_df, _ = self.fd.connected_components()
        clusters_df = get_clusters(connected_components_df)
        cluster_images_to_keep = []
        list_of_duplicates = []

        for cluster_file_list in clusters_df.filename:
            # keep first file, discard rest
            keep = cluster_file_list[0]
            discard = cluster_file_list[1:]

            cluster_images_to_keep.append(keep)
            list_of_duplicates.extend(discard)

        print(f"Found {len(set(list_of_duplicates))} highly similar images to discard")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(self.remove_images, list_of_duplicates), total=len(list_of_duplicates)))

    def copy_broken_image(self, img_path):
        name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(self.result, 'results', 'broken_images', name))

    def copy_duplicate_image(self, img_path):
        name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(self.result, 'results', 'duplicate_images', name))

    def copy_outliers_image(self, img_path):
        name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(self.result, 'results', 'outliers_images', name))

    def copy_retrieval_image(self, img_path):
        name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(self.result, 'results', 'retrieval_images', name))

    def remove_images(self, img_path):
        os.remove(img_path)

    def vis_images(self, ):
        fastdup.create_duplicates_gallery(os.path.join(self.result, 'work_dir', 'similarity.csv'),
                                          save_path=os.path.join(self.result, 'vis'))
        fastdup.create_outliers_gallery(os.path.join(self.result, 'work_dir', 'outliers.csv'),
                                        save_path=os.path.join(self.result, 'vis'))
        fastdup.create_components_gallery(os.path.join(self.result, 'work_dir', 'connected_components.csv'),
                                          save_path=os.path.join(self.result, 'vis'))

    def vis_culster_images(self):
        if self.is_dinov2:
            dim = 768
        else:
            dim = 576
        filenames, feature_vec = fastdup.load_binary_feature(
            os.path.join(self.result, 'work_dir', 'atrain_features.dat'), d=dim)
        connected_components_df = pd.read_csv(os.path.join(self.result, 'work_dir', 'connected_components.csv'))
        component_id = connected_components_df['component_id'].to_numpy()
        vis_cluster_tsne(feature_vec, component_id, filenames, self.result)

    def data_processing(self, num):
        if not os.path.exists(os.path.join(self.result, 'results', 'outliers_images')):
            os.makedirs(os.path.join(self.result, 'results', 'outliers_images'))
        if not os.path.exists(os.path.join(self.result, 'results', 'broken_images')):
            os.makedirs(os.path.join(self.result, 'results', 'broken_images'))
        if not os.path.exists(os.path.join(self.result, 'results', 'duplicate_images')):
            os.makedirs(os.path.join(self.result, 'results', 'duplicate_images'))

        self.find_outlier_image(num=num)
        self.find_duplicate_images(num=num)
        self.find_borken_images(num=num)

    def retrieval(self, cluster_num, query_image_path):
        if not os.path.exists(os.path.join(self.result, 'results', 'retrieval_images')):
            os.makedirs(os.path.join(self.result, 'results', 'retrieval_images'))
        self.find_retrieval_images(cluster_num, query_image_path)

    def vis_processing(self):
        self.vis_images()
        self.vis_culster_images()

    def delete_duplicate_images(self):
        self.remove_duplicate_images()