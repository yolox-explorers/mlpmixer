import ast
import pandas as pd

class Clean:
    """Cleans issues detected during data checks and provide clean dataframe for modeling.
    """
    def __init__(self, raw_pickle_path: str):
        """Initializes class Clean.

        Args:
            raw_pickle_path (str): absolute filepath of raw pickle file.
        """
        self.raw_pickle_path = raw_pickle_path

    def get_dataframe(self):
        """Gets information in pickle file read into dataframe.

        Returns:
            df (pd.DataFrame): output dataframe where each row contains information for an image.
        """
        df = pd.read_pickle(self.raw_pickle_path)

        return df

    def fill_missing(self, df: pd.DataFrame):
        """Fill missing values in:
        - `time_of_day` column that are not `night`, `morning`, or `afternoon`,
            with `day` value.
        - `weather_condition` column with `mix` value.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe 
        """
        df["time_of_day"] = df["time_of_day"].fillna("day")
        df["weather_condition"] = df["weather_condition"].fillna("mix")

        return df

    def get_occlusion_label(self, df: pd.DataFrame):
        """Get occlusion boolean label for car and obstacle from annotations,
        saved to 2 new columns `car_is_occluded` and `obstacle_is_occluded`.
        Includes (and counts number of) background images without car label
        but with obstacle label.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe with 2 new columns
        """
        car_occluded_list, obstacle_occluded_list, background_count = [], [], 0
        for i in range(len(df)):
            annot = df["annotations"].iloc[i]

            car_annot = annot["Motorcycle"]
            obstacle_annot = annot["Obstacle"]

            if len(car_annot) != 0:
                # only for non-background images with car annotations
                car_is_occluded = car_annot[0]["attributes"]["occluded"]
            else:
                background_count += 1
                car_is_occluded = None
            obstacle_is_occluded = obstacle_annot[0]["attributes"]["occluded"]

            car_occluded_list.append(car_is_occluded)
            obstacle_occluded_list.append(obstacle_is_occluded)

        print(f"No. of background images: {background_count}")
        df["car_is_occluded"] = car_occluded_list
        df["obstacle_is_occluded"] = obstacle_occluded_list

        return df

    def correct_occlusion_label(self, df: pd.DataFrame, wrong_label: dict):
        """Amends wrong occlusion label during annotation by specifying 
        folder name and image names under `wrong_label` dictionary into
        four sub-categories:
        - `obs_occ` (obstacle labelled as occluded but is not),
        - `obs_not_occ` (obstacle labelled as not occluded but is),
        - `motor_occ` (car labelled as occluded but is not),
        - `motor_not_occ` (car labelled as not occluded but is).
        For folders containing majority mislabelled and few correct occlusion, 
        add the latter to `true_occlusion` sub-category.

        Args:
            df (pd.DataFrame): input dataframe
            wrong_label (dict): dictionary in config file containing 
                sub-categories (dictionary) containing folder name (key) 
                and image name(s) (value).

        Returns:
            df (pd.DataFrame): output dataframe with corrected occlusion labels.
        """

        discrepant = list(df[(df['description'] == list(wrong_label["true_occlusion"].keys())[0]) \
            & (df['car_is_occluded'] == True)]['img_fname'])
        discrepant_motor = [
            i for i in discrepant if i not in list(wrong_label["true_occlusion"].values())[0]
        ]

        for i in range(len(df)):
            # obstacle labelled as occluded but is not
            for k, v in wrong_label["obs_occ"].items():
                if (df['description'].iloc[i] == k) & (df['img_fname'].iloc[i] in v):
                    df['obstacle_is_occluded'].iloc[i] = False

            # obstacle labelled as not_occluded but is occluded
            for k, v in wrong_label["obs_not_occ"].items():
                if (df['description'].iloc[i] == k) & (df['img_fname'].iloc[i] in v):
                    df['obstacle_is_occluded'].iloc[i] = True

            # car labelled as occluded but is not
            for k, v in wrong_label["true_occlusion"].items():
                if (df['description'].iloc[i] == k) & (df['img_fname'].iloc[i] in discrepant_motor):
                    df['car_is_occluded'].iloc[i] = False

            for k, v in wrong_label["motor_occ"].items():
                if (df['description'].iloc[i] == k) & (df['img_fname'].iloc[i] == v):
                    df['car_is_occluded'].iloc[i] = False

            # car labelled as non_occluded but is occluded
            for k, v in wrong_label["motor_not_occ"].items():
                if (df['description'].iloc[i] == k) & (df['img_fname'].iloc[i] in v):
                    df['car_is_occluded'].iloc[i] = True

        return df

    def remove_duplicate_annotations(self, df: pd.DataFrame):
        """Removes duplicate annotations in car and obstacle if present.
        Counts and prints image filename for every duplicate instance found.
        Counts total number of images containing at least 1 duplicate.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe without duplicate annotations
        """
        new_annot_list = []
        n_count = 0
        for i in range(len(df)):
            annot = df["annotations"].iloc[i] # refer to example annotations below
            new_annot = {}
            bool_dup = False

            for k, v_list in annot.items():     # key k is either car or obstacle
                new_v_list = []

                if len(v_list) == 1: # ie. only 1 segmentation mask for the key k
                    new_v_list = v_list
                elif len(v_list) > 1: # ie. more than 1 segmentation mask for the key k
                    for seg in v_list:
                        if seg not in new_v_list:
                            new_v_list.append(seg)  # add only unique segmentations to list
                        else:
                            bool_dup = True # duplicate annotation is found, but not added to list
                            print(f"Image: {df['img_fpath'].iloc[i]}")

                new_annot[k] = new_v_list   # assigns new annotations list for the key k

            new_annot_list.append(new_annot)    # assigns new annotations for all the keys in this image(row)
            if bool_dup is True:
                n_count += 1

        df["annotations"] = new_annot_list
        print(f"No. of images with at least 1 duplicate annotation: {n_count}")

        return df

    def delete_empty_annotations(self, df: pd.DataFrame):
        """Deletes empty segmentation annotations (ie. empty segmentation list) if present.
        Counts the number of empty segmentations per car/obstacle.
        Counts the total number of images with at least 1 empty segmentation.
        Assumption: all the target car and obstacles have been correctly labelled, and
        any empty segmentation list is a result of extra erroneous data labelling.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe without empty annotations
        """
        n_count = 0
        new_annot_list = []
        for i in range(len(df)):
            annot = df["annotations"].iloc[i]
            n_empty = 0
            bool_empty = False

            new_annot = {}
            for k, v_list in annot.items():     # key k is either car or obstacle
                if len(v_list) != 0:
                    for seg in v_list:
                        if len(seg["segmentation"]) == 0:   # segmentation for key k is an empty list
                            print(f"Before: \nLength of {k}: ({len(v_list)}) ---> {v_list}")
                            v_list.remove(seg)
                            n_empty += 1
                            bool_empty = True
                            print(f"After: \nLength of {k}: ({len(v_list)}) ---> {v_list} \
                                    \nImage: {df['img_fpath'].iloc[i]} \
                                    \nNo. of empty annotations removed: {n_empty}\n")

                new_annot[k] = v_list

            if bool_empty is True:
                n_count += 1
                new_annot_list.append(new_annot)
            else:
                new_annot_list.append(annot)

        df["annotations"] = new_annot_list
        print(f"No. of images with at least 1 empty annotation: {n_count}")

        return df

    def remove_extra_annotation_category(self, df: pd.DataFrame):
        """Removes extra park annotation category which is not in use in 
        `annotations` as well as in `annotation_categories`.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe without park annotation category
        """
        # remove from annotations
        n_count = 0
        new_annot_list = []
        for i in range(len(df)):
            annot = df["annotations"].iloc[i]
            if "Park" in annot.keys():
                annot.pop("Park")
                n_count += 1
            new_annot_list.append(annot)
        df["annotations"] = new_annot_list        

        # remove from annotation categories
        print(f"Before: \n{df['annotation_categories'].value_counts()}")
        df["annotation_categories"] = df["annotation_categories"].astype("string")
        df["annotation_categories"] = df["annotation_categories"].replace(
            "['Motorcycle', 'Plank', 'Obstacle']",
            "['Motorcycle', 'Obstacle']"
        )
        df["annotation_categories"] = df["annotation_categories"].replace(
            "['Motorcycle', 'Obstacle']",
            "['Car', 'Obstacle']"
        ).apply(ast.literal_eval) # converts string in list to nonstring
        print(f"After: \n{df['annotation_categories'].value_counts()}")

        return df

    def reverse_annotation_category(self, df: pd.DataFrame):
        """Reverses the annotations as well as the annotation category,
        from [Obstacle, Motorcycle] to [Motorcycle, Obstacle].

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe with reversed annotations and annotation category
        """
        print(f"Before: \n{df['annotation_categories'].value_counts()}")
        # reverse annotations
        rev_annots_list = []
        for i in range(len(df)):
            if df["annotation_categories"].iloc[i] == "['Obstacle', 'Motorcycle']":
                annots = df["annotations"].iloc[i]
                rev_annots = dict(reversed(list(annots.items())))
                rev_annots_list.append(rev_annots)
            else:
                rev_annots_list.append(df["annotations"].iloc[i])

        df["annotations"] = rev_annots_list

        # reverse annotation category
        df["annotation_categories"] = df["annotation_categories"].astype("string")
        df["annotation_categories"] = df["annotation_categories"].replace(
            "['Obstacle', 'Motorcycle']",
            "['Car', 'Obstacle']"
        ).apply(ast.literal_eval)
        df["annotation_categories"] = df["annotation_categories"].astype("object")
        print(f"After: \n{df['annotation_categories'].value_counts()}")

        return df

    def export_files(self, df: pd.DataFrame, clean_fpath: str):
        """Exports clean dataframe to a pickle and a csv file.

        Args:
            df (pd.DataFrame): clean input dataframe
            clean_fpath (str): relative filepath EXCLUDING EXTENSIONS for clean pickle & csv files
        """
        df.to_pickle(clean_fpath + ".pkl")
        df.to_csv(clean_fpath + ".csv")
        print(f"Clean pickle and csv files exported to: {clean_fpath}")
