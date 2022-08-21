from itertools import count
import os
import glob
import json
import regex as re
from datetime import datetime
import patoolib
from shutil import copyfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import imagehash
import hydra
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil

from src.modelling.data_prep.clean_data import Clean

class PrepData:
    def __init__(self, args):
            self.img_zip_dir = os.path.abspath(args["data_prep"]["img_zip_dir"])
            self.img_save_dir = os.path.abspath(args["data_prep"]["img_save_dir"])
            self.pickle_exists = os.path.exists(os.path.join(self.img_save_dir,'img_data.pkl'))
            self.val_split = args["train"]["val_split"]
            self.test_split = args["train"]["test_split"]

            self.station_list = ['slope','park']
            self.weather_list = ['heavy','light','cloudy','rain','sunny', 'clear']
            self.time_of_day_list = ['morning','afternoon','night','evening','day']

            self.data = None
            self.hash_similarity_matrix = None
            print(self.img_zip_dir)
            print(self.img_save_dir)
            print(self.pickle_exists)
    
    def __extract_L2_zip(self,dir):
        """
        Extracts level 2 (ie. .zip files within a .zip file) .zip files and deletes the original .zip file

        Args:
            dir ([str]): Directory where the .zip files may be located.
        """

        for fpath in glob.glob(dir + "/**/*.[zr][ia][pr]", recursive=True):
            dest_fpath_dir = fpath.split('.')[0]
            os.makedirs(dest_fpath_dir, exist_ok=True)
            print(f"Created directory: \"{dest_fpath_dir}\"")
            patoolib.extract_archive(fpath, outdir=dest_fpath_dir)
            os.remove(fpath)


    def extract_data(self):
        """
        Method to copy/extract all data from given directory to a new directory, while maintaining the same folder structure. 
        - Will copy all files except .zip files.
        - Will extract all .zip files to a destination folder folllowing the name of the .zip file.
        - Can be used to update destination directory if there are updated image zip files.

        Args:
            src_dir (str, optional): Source directory containing the original data including .zip files i.e. '.\data\cleaned'.
            dest_dir (str, optional): Destination directory to copy all files. Defaults to '.\data\extracted'.
        """

        print('Extracting zip files...')

        src_dir = self.img_zip_dir
        dest_dir = self.img_save_dir

        count_copied, count_zip, count_exist = 0, 0, 0
        for root, dirs, files in os.walk(src_dir):

            for file in files:

                src_fpath = os.path.join(root, file)

                # For extracting zip & rar files
                if src_fpath.endswith('.zip') or src_fpath.endswith('.rar'):
                    dest_fpath_dir = os.path.join(dest_dir, os.path.relpath(src_fpath, src_dir)).split('.')[0]

                    if os.path.exists(dest_fpath_dir):
                        print(f'{dest_fpath_dir} Already exists!')

                    else:
                        if not os.path.exists(dest_fpath_dir):
                            os.makedirs(dest_fpath_dir, exist_ok=True)
                            print(f"Created directory: \"{dest_fpath_dir}\"")

                        patoolib.extract_archive(src_fpath, outdir=dest_fpath_dir)

                        self.__extract_L2_zip(dest_fpath_dir)
                        count_zip+=1

                # For extracting non zip files
                else:
                    dest_fpath = os.path.join(dest_dir, os.path.relpath(src_fpath, src_dir))
                    dest_fpath_dir = os.path.dirname(dest_fpath)

                    if os.path.exists(dest_fpath):
                        count_exist+=1
                    else:
                        if not os.path.exists(dest_fpath_dir):
                            os.makedirs(dest_fpath_dir, exist_ok=True)
                            print(f"Created directory: \"{dest_fpath_dir}\"")

                        copyfile(src_fpath,dest_fpath)
                        count_copied+=1

        self.no_changes = (count_zip+count_copied)==0

        print(f'{count_zip} zip/rar files extracted. {count_copied} new files (non .zip/.rar) copied. {count_exist} files (non .zip/.rar) already exist.')
        if self.no_changes:
            print('No data to extract.')
        else:
            print(f'Data extracted to {dest_dir}')
    
    def __infer_upload_date(self, img_fpath):
        """
        Infers the date from the given filepath string.
        """
        date_string = re.findall('\d+_\d+', img_fpath)[0]
        date = datetime.strptime(date_string[:8], '%Y%m%d')
        return date

    def __populate_images(self):
        """
        Scans the directory for all .jpg and .jpeg images and returns a dataframe.

        Args:
            data_dir (str, optional): Directory for all images. Defaults to '.\data\extracted'.

        Returns:
            pd.DataFrame: Dataframe containing img_fpath, img_fname and img_dir.
        """
        data_dir = self.img_save_dir

        img_fpaths = []
        for root, dirs, files in os.walk(data_dir):
            img_fpaths += [
                os.path.join(root,file) for file in files if \
                    file.endswith('.jpg') or \
                    file.endswith('.jpeg')
            ]

        data = pd.DataFrame({'img_fpath':img_fpaths})
        data['upload_date'] = data['img_fpath'].apply(self.__infer_upload_date)
        data['img_fname'] = data['img_fpath'].apply(os.path.basename)
        data['img_dir'] = data['img_fpath'].apply(os.path.dirname)

        return data

    def __get_image_detail_string(self, img_fpath):
        """
        Returns the correct folder of the image which contains its description.
        Looks for the first folder in which the image filepath resides, then looks at the next lower folder. 
        """
        _dir = os.path.dirname(img_fpath).lower()
        for i in range(2):
            for station in self.station_list:
                if station in os.path.basename(_dir):
                    break
            else:   # only excecuted if the inner loop did NOT break
                _dir = os.path.dirname(_dir)
                details = None
                continue
            details = os.path.basename(_dir)
            break   # only excecuted if the inner loop DID break
        return details
    
    def __infer_sample(self, txt):
        """
        Infers if the image is a sample image.
        """

        def is_sample(txt):
            if txt == None:
                # return 'Unknown'
                return None
            elif 'sample' in txt.lower():
                # return "True"
                return True
            elif 'example' in txt.lower():
                # return "True"
                return True
            else:
                # return "False"
                return False
        return str(is_sample(txt)) if is_sample(txt) != None else "Unknown"

    def __infer_station(self, txt):
        """
        Infers the station.
        """
        if txt == None:
            return None
        for station in self.station_list:
            if station in txt.lower():
                return station
    
    def __infer_pass_fail(self, txt):
        """
        Infers the pass or fail condition.
        """
        if txt == None:
            return None
        elif 'pass' in txt.lower():
            return 'pass'
        elif 'fail' in txt.lower():
            return 'fail'
        elif 'on' in txt.lower():
            return 'on'
        elif 'off' in txt.lower():
            return 'off'
        else:
            return None
    
    def __infer_weather(self, txt):
        """
        Infers the weather condition.
        """
        if txt == None:
            return None
        condition = []
        for weather in self.weather_list:
            if weather in txt.lower():
                condition.append(weather)
        if len(condition)==0:
            return None
        else:
            condition = sorted(condition, key=lambda cond: self.weather_list.index(cond))
            return '_'.join(condition)

    def __infer_time_of_day(self, txt):
        """
        Infers the time of day
        """
        if txt == None:
            return None
        condition = []
        for tod in self.time_of_day_list:
            if tod in txt.lower():
                condition.append(tod)
        if len(condition)==0:
            return None
        else:
            condition = sorted(condition, key=lambda cond: self.time_of_day_list.index(cond))
            return '_'.join(condition)

    def __find_annotations_fpath(self,img_dir):
        """
        Searched for the corresponding annotations file path for the given image filepath.
        """
        search_dir_1 = img_dir
        search_dir_2 = os.path.join(os.path.dirname(img_dir),'annotations')

        search_1 = [file for file in os.listdir(search_dir_1) if file.endswith('.json')]
        try:
            search_2 = [file for file in os.listdir(search_dir_2) if file.endswith('.json')]
        except:
            search_2 = []

        if len(search_1)==1:
            return os.path.join(search_dir_1,search_1[0])
        elif len(search_2)==1: 
            return os.path.join(search_dir_2,search_2[0])
        else:
            return None
    
    def __read_json(self,s):
        """
        Reads the annotations file and extracts the width, height, annotation_categories and annotations.
        """

        if s['annotations_fpath']==None:
            width, height, annotation_categories, annotations = 0, 0, [], None
        else:
            f = open(s['annotations_fpath'])
            json_data = json.load(f)
            categories = {x['id']:x['name'] for x in json_data['categories']}
            img_id, width, height = [(x['id'],x['width'],x['height']) for x in json_data['images'] if x['file_name'] == s['img_fname']][0]
            annot_keys = ['segmentation','area','bbox','attributes']
            annotations = {categories[x]:[{z:y[z] for z in y.keys() if z in annot_keys} for y in json_data['annotations'] \
                if y['category_id']==x and y['image_id']==img_id] for x in categories.keys()}
            annotation_categories = annotations.keys()
        s['width'] = width
        s['height'] = height
        s['annotation_categories'] = list(annotation_categories)
        s['annotations'] = annotations
        return s

    def __get_image_hash(self, img_fpath, hash_size=32):
        """Calculates the image average hash. 

        Args:
            img_fpath (string): Image filepath

        Returns:
            string: Image hash
        """
        img = Image.open(img_fpath)
        hash = imagehash.average_hash(img, hash_size=hash_size)
        return hash
        
    def get_data(self):
        """Generates a pandas DataFrame with all the related image data and annotations.

        Returns:
            DataFrame: Image data and annotations
        """
        if self.no_changes and self.pickle_exists:
            data = pd.read_pickle(os.path.join(self.img_save_dir,'img_data.pkl'))
        else:
            data = self.__populate_images()
            data['description'] = data['img_fpath'].apply(self.__get_image_detail_string)
            data['is_sample'] = data['description'].apply(self.__infer_sample)
            data['station'] = data['description'].apply(self.__infer_station)
            data['pass_fail'] = data['description'].apply(self.__infer_pass_fail)
            data['weather_condition'] = data['description'].apply(self.__infer_weather)
            data['time_of_day'] = data['description'].apply(self.__infer_time_of_day)
            data['annotations_fpath'] = data['img_dir'].apply(self.__find_annotations_fpath)
            tqdm.pandas(desc='Retrieving image annotations')
            data.append(['width','height','annotation_categories','annotations'])
            data = data.progress_apply(self.__read_json,axis=1)
            data['width'] = data['width'].astype(int)
            data['height'] = data['height'].astype(int)
        self.data = data
        return data
    
    def compute_image_hash(self, data):
        """
        Appends the image hash to the input dataframe.

        Returns:
            DataFrame
        """
        if 'avg_hash' in data.columns:
            return data['avg_hash']
        else:
            tqdm.pandas(desc='Calculating image hash')
            data['avg_hash'] = data['img_fpath'].progress_apply(self.__get_image_hash)
            return data['avg_hash']
    
    def __compute_hash_similarity_matrix(self, avg_hash):
        """
        Computes a similarity matrix by calculating the difference of hash values of each image pair.
        """
        if self.hash_similarity_matrix is None:
            hashes = avg_hash.to_numpy()
            matrix = [[abs(i[1]-j[1]) if i[0]>j[0] else 0 for i in enumerate(hashes)] for j in enumerate(tqdm(hashes, desc = 'Generating hash similarity matrix'))]
            int32_max = np.iinfo(np.int32).max
            matrix = np.array(matrix, dtype=np.int32)
            matrix = matrix + np.transpose(matrix) + np.diag(np.diag(np.ones(matrix.shape)*int32_max))
            self.hash_similarity_matrix = matrix
        else:
            pass

    def compute_most_similar_image(self, avg_hash):
        """
        Fills the dataframe with the image index which is most similar.
        """
        if self.data is None or 'most_similar_image' not in self.data.columns:
            self.__compute_hash_similarity_matrix(avg_hash)
            index = np.argmin(self.hash_similarity_matrix, axis=0)
            return index
        else:
            return self.data['most_similar_image']
    
    def compute_hash_min_difference(self, avg_hash):
        """
        Fills the dataframe with the most similar image hash difference
        """
        if self.data is None or 'min_hash_diff' not in self.data.columns:
            self.__compute_hash_similarity_matrix(avg_hash)
            min_diff = self.hash_similarity_matrix.min(axis=0)
            return min_diff
        else:
            return self.data['min_hash_diff']

    
    def train_val_test_split(self, dataframe):
        """
        Groups dataframe by station, time_of_day, on_off, weather_condition and sorts by avg hash
        Splits into 80% train, 10% val, 10% test set
        """

        dataframe["avg_hash"] = dataframe.apply(lambda row: int(str(row["avg_hash"]), \
            base=16), axis=1)

        dataframe = dataframe.fillna({'pass_fail': "None"})
        # Get proportion count
        df = dataframe.groupby(['station', 'time_of_day', 'pass_fail', \
            'weather_condition']).size().reset_index()

        dataframe = dataframe.sort_values(by = ['station', 'time_of_day', 'pass_fail', \
            'weather_condition', 'avg_hash'])
        dataframe['train_val_test'] = None

        data = pd.DataFrame()
        for i in range(len(df)):
            station = df['station'].iloc[i]
            time_of_day = df['time_of_day'].iloc[i]
            on_off = df['pass_fail'].iloc[i]
            weather = df['weather_condition'].iloc[i]

            temp_df = dataframe.loc[(dataframe['station'] == station) & \
                (dataframe['time_of_day'] == time_of_day) & \
                (dataframe['pass_fail'] == on_off) & \
                (dataframe['weather_condition'] == weather)].sort_values('avg_hash')
            
            val_index = int(len(temp_df) * (1 - self.val_split - self.test_split)) # 80% percentile
            test_index = int(len(temp_df) * (1 - self.test_split)) # 90% percentile
            
            train_df = temp_df.iloc[:val_index]
            train_df.loc[:, 'train_val_test'] = 'train'

            val_df = temp_df.iloc[val_index: test_index]
            val_df.loc[:, 'train_val_test'] = 'val'

            test_df = temp_df.iloc[test_index:]
            test_df.loc[:, 'train_val_test'] = 'test'

            data = data.append([train_df, val_df, test_df])

        print('output dataframe: ', data.shape)
        print('input dataframe: ', dataframe.shape)

        return data


    def gen_images(self, image_catalog_path, stations, times, data_sets, img_save_dir):
        """
        Generates a csv file of all the images.
        """

        if image_catalog_path is None:
            raise Exception("Please specify a image_catalog_path")

        df = pd.read_pickle(image_catalog_path)

        result = pd.DataFrame(columns=['station', 'time', 'data_set', 'pass_fail', 'count'])
        
        for station in stations:
            for time in times:
                for data_set in data_sets:
                    for pass_fail in ["on", "off", 'None']:

                        data = df[df["station"] == station]

                        if "night" == time:
                            data = data[data["time_of_day"] == "night"]
                        else:
                            data = data[data["time_of_day"] != "night"]

                        data = data[data['train_val_test'] == data_set]
                        data = data[data['pass_fail'] == pass_fail]

                        for _, x in data.iterrows():
                            target_path = img_save_dir + os.path.sep + station + os.path.sep + time + os.path.sep + data_set + os.path.sep + (pass_fail.upper() if pass_fail != "None" else "OFF") + os.path.sep + x["upload_date"].strftime("%Y-%m-%d")  + "_" + x["description"]  + "_" + x["img_fname"] 
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)

                            shutil.copy(x["img_fpath"], target_path)

                            print(x["img_fpath"] + " -> " + target_path)

                        row = {'station': station, 'time': time, 'data_set': data_set, 'pass_fail': "off (background)" if pass_fail == "None" else pass_fail, 'count': len(data)}
                        result = result.append(row, ignore_index=True)

        print(result)
        result.to_csv(os.path.join(img_save_dir,"gen_images.csv"))

@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    # copy cleaned folder and sub-folders from mount point to OS disk (if applicable)
    if args["data_prep"]["mount_path"] is not None:
        print(f"\nCopying folder contents \
                \nfrom: {args['data_prep']['mount_path']} \
                \nto: {os.path.abspath(args['data_prep']['img_zip_dir'])}")
        shutil.copytree(
            src=args["data_prep"]["mount_path"],
            dst=os.path.abspath(args["data_prep"]["img_zip_dir"])
        )

    testing = args["data_prep"]["testing"]
    prep = PrepData(args)
    print(args)

    if not testing:
        gen_images = args["data_prep"]["gen_images"]
        if gen_images:
            prep.gen_images(args["data_prep"]["image_catalog_path"], args["data_prep"]["stations"], args["data_prep"]["times"], args["data_prep"]["data_sets"], args["data_prep"]["img_save_dir"])
            return 

        prep.extract_data()
        data = prep.get_data()
        
        data.to_pickle(os.path.join(prep.img_save_dir,'img_data.pkl'))
        data.to_csv(os.path.join(prep.img_save_dir,"img_data.csv"))

    else:
        data = pd.read_pickle(os.path.join(prep.img_save_dir,'img_data.pkl'))
        print('First five rows:')
        print(data.head())
        print('\nInfo:')
        print(data.info())
        print(f'\nAnnotation type: {type(data["annotations"].iloc[100])}')

    clean = Clean(
        raw_pickle_path = os.path.join(prep.img_save_dir,'img_data.pkl')
    )
    df = clean.get_dataframe()
    df = clean.fill_missing(df = df)
    df = clean.remove_duplicate_annotations(df = df)
    df = clean.remove_extra_annotation_category(df = df)
    df = clean.reverse_annotation_category(df = df)
    df = clean.delete_empty_annotations(df = df)
    # apply conditional if using occlusion label
    if args["data_prep"]["occlude"]:
        df = clean.get_occlusion_label(df = df)
        df = clean.correct_occlusion_label(
            df = df,
            wrong_label = args["data_prep"]["wrong_label"]
        )

    df['avg_hash'] = prep.compute_image_hash(df)
    df['most_similar_image'] = prep.compute_most_similar_image(df['avg_hash'])
    df['min_hash_diff'] = prep.compute_hash_min_difference(df['avg_hash'])
    df = prep.train_val_test_split(df)

    print(df.info())
    print(df.head())
    print(df["annotation_categories"].value_counts())
    print(df['train_val_test'].value_counts(normalize = True))
    clean.export_files(
        df = df,
        clean_fpath = os.path.join(prep.img_save_dir,'clean_img_data')
    )

if __name__ == '__main__':
    main()
