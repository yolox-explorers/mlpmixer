import argparse
import ast
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

class Check:
    def __init__(self, df: pd.DataFrame):
        """Initializes class Check with input dataframe.

        Args:
            df (pd.DataFrame): input dataframe
        """
        self.df = df

    def get_img_channels(self, df: pd.DataFrame):
        """Loops through images to obtain and checks:
            - if image shape matches height and width in pickle file
            - if image shape is of length 3 (ie. height, width, channel only)
            - appends image channel value to list
        - adds list to new dataframe column `img_channels`

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe with new column `img_channels`
        """
        n = len(df)
        n_wrong_size = 0
        n_wrong_length = 0
        img_channel_list = []

        print("Generating and checking image colour channels...")
        for i in tqdm(range(0, n)):
            img_fpath = df["img_fpath"].iloc[i]
            img = cv2.imread(img_fpath)

            # check if shape match height, width in pickle file
            height = df["height"].iloc[i]
            width = df["width"].iloc[i]
            if height != img.shape[0] or width != img.shape[1]:
                n_wrong_size += 1
                print(
                    f"Image: {img_fpath} \nShape: {img.shape} \
                    \nHeight, Width: {height, width}"
                )

            # check if shape is length of 3
            if len(img.shape) != 3:
                n_wrong_length += 1
                print(f"Image: {img_fpath} \nWrong shape: {img.shape}")

            # append channels to list
            img_channel_list.append(img.shape[2])

        # add new column img_channels
        df["img_channels"] = img_channel_list
        print(f"No. of images with wrong size: {n_wrong_size}")

        return df

    def counts(self, df: pd.DataFrame, groupby_list: list):
        """Prints counts by:
        - station
        - weather condition
        - image size(height, width)
        - colour channels
        - annotation categories
        - pass/fail status

        Args:
            df (pd.DataFrame): input dataframe
            groupby_list (list): list of lists of dataframe columns to group by
        """
        for i in groupby_list:
            if "annotation_categories" in i:
                df["annotation_categories"] = df["annotation_categories"].astype("string")
            print(f"{df.groupby(by=i).count().iloc[:, :1]}\n\n")

    def high_level_counts(self, df: pd.DataFrame):
        """Counts on a high level, by each station, the number of:
        - images
        - annotations for car
        - annotations for obstacle(s)

        Args:
            df (pd.DataFrame): input dataframe
        """
        station_list = list(df["station"].unique())

        for i in station_list:
            df_station = df[df["station"] == i].reset_index(drop=True)
            num_imgs = len(df_station) # store this
            car_annot_list, obstacle_annot_list = [], []

            for j in tqdm(range(0, num_imgs)):
                annot = df_station["annotations"].iloc[j]
                car_annot = annot["Car"]
                obstacle_annot = annot['Obstacle']
                car_annot_list.append(len(car_annot))
                obstacle_annot_list.append(len(obstacle_annot))
            num_cars = sum(car_annot_list) # store this
            num_obstacles = sum(obstacle_annot_list) # store this

            if i == "park":
                x = 1
            elif i == "slope":
                x = 9
            print(f"\n############# High-level counts ############# \
                    \nStation: {i} \
                    \nNo. of images: {num_imgs} \
                    \nNo. of cars: {num_cars} \
                    \nNo. of obstacles: {num_obstacles} (Expected: {x*num_imgs}) \
                    \n############# End #############\n\n")

    def occlusion_label(self, df: pd.DataFrame):
        """Checks and prints counts of occlusions:
        - for day images:
            - all car and obstacle occlusion should be False
        - for night (park) images:
            - checks if filename occlusion label matches CAR occlusion
                label, else counts towards `wrong_occ`
            - counts of car occluded / not occluded
            - counts of obstacle occluded / not occluded

        Args:
            df (pd.DataFrame): input dataframe containing 
                `car_is_occluded` and `obtacle_is_occluded` columns
        """
        car_is_occ, car_not_occ, wrong_occ, \
            obstacle_is_occ, obstacle_not_occ, \
                day_obs_notocc, day_car_notocc = 0, 0, 0, 0, 0, 0, 0
        night_df = df[df['time_of_day'] == 'night']
        desc_bool = None    # description from folder name
        for i in tqdm(range(len(df))):
            if df["time_of_day"].iloc[i] == "night":
                if ("not_occluded" in df["description"].iloc[i]) |\
                    ("notoccluded" in df["description"].iloc[i]):
                    desc_bool = False
                elif "occluded" in df["description"].iloc[i]:
                    desc_bool = True
            else:
                desc_bool = None

            if df["obstacle_is_occluded"].iloc[i] == True:
                obstacle_is_occ += 1
            elif df["obstacle_is_occluded"].iloc[i] == False:
                if df["time_of_day"].iloc[i] == "night":
                    obstacle_not_occ += 1
                else:
                    day_obs_notocc += 1

            if desc_bool == df["car_is_occluded"].iloc[i] == True:
                car_is_occ += 1
            elif desc_bool == df["car_is_occluded"].iloc[i] == False:
                car_not_occ += 1
            elif desc_bool is None and \
                df["car_is_occluded"].iloc[i] == False and \
                    df["time_of_day"].iloc[i] != "night":
                day_car_notocc += 1
            elif desc_bool != df["car_is_occluded"].iloc[i]:
                wrong_occ += 1

                print(f"\nImage: {df['img_fpath'].iloc[i]} \
                        \nFilename Description: {desc_bool} \
                        \nCar is occluded: {df['car_is_occluded'].iloc[i]} \
                        \nObstacle is occluded: {df['obstacle_is_occluded'].iloc[i]}")

        print(f"Summary: \
                \nTotal no. of images: {len(df)} \
                \n\nTotal no. of night images: {len(night_df)} \
                \nNo. of car is occluded (correct): {car_is_occ} \
                \nNo. of car is not occluded (correct): {car_not_occ} \
                \nNo. of obstacle is occluded: {obstacle_is_occ} \
                \nNo. of obstacle is not occluded: {obstacle_not_occ} \
                \nNo. of wrong occlusion labels: {wrong_occ} \
                \n\nTotal no. of day images: {len(df) - len(night_df)} \
                \nNo. of car is not occluded: {day_car_notocc} \
                \nNo. of obstacle is not occluded: {day_obs_notocc}")

    def annotation_counts(self, df: pd.DataFrame, station: str):
        """Checks if car and obstacle counts = 1 (or 9 for slope).
        Prints out entries that do not fulfil this criteria for further checks:
        - duplicate annotations (ie. exact segmentation mask duplicates for 
            { segmentation, area, bbox } dictionary entry, with and without extra annotations)
        - empty annotations (ie. empty list for segmentation value, with some value for area & bbox)
        - missing annotations (ie. no segmentation & area & bbox entry)
        - extra annotations (ie. duplicate/different segmentation/area/bbox values)
        - number of slope obstacle annotations below default value of 9

        Args:
            df (pd.DataFrame): input dataframe
            station (str): type of station, either "park" or "slope"

        Returns:
            list_wrong (list): dataframe rows with wrong annotation counts
        """
        df_station = df[df["station"] == station].reset_index(drop=True)

        n = len(df_station)
        n_wrong, n_dups, n_dup_extra, n_empty, \
            n_missing, n_extra, n_below, list_wrong = 0, 0, 0, 0, 0, 0, 0, []

        for i in tqdm(range(0, n)):
            annot = df_station["annotations"].iloc[i]
            car_annot = annot["Car"]
            obstacle_annot = annot['Obstacle']
            if station.lower() == "park":
                if len(annot["Car"]) != 1 or \
                    len(annot["Obstacle"]) != 1:
                    n_wrong += 1
                    list_wrong.append(i)

                    bool_dup_extra, bool_dup = self._duplicate_annots(
                        station, car_annot, obstacle_annot
                    )
                    bool_empty = self._empty_annots(annotation=annot)
                    bool_missing = self._missing_annots(annotation=annot)

                    if bool_dup is True:
                        n_dups += 1

                    if bool_dup_extra is True:
                        n_dup_extra += 1

                    if bool_empty is True:
                        n_empty += 1

                    if bool_missing is True:
                        n_missing += 1

                    if (bool_dup is False and bool_empty is False and \
                        bool_missing is False) and \
                        (len(annot["Car"]) > 1 or len(annot["Obstacle"]) > 1):
                        print("No empty/missing/duplicate annotations. \
                                \nManually check for extra annotations.")
                        self._extra_annots(car_annot, obstacle_annot)
                        n_extra += 1

                    print(f"\nNo. of Cars: {len(annot['Car'])} \
                            \nNo. of Obstacles: {len(annot['Obstacle'])} \
                            \nCar Annotation: {annot['Car']} \
                            \nObstacle Annotation: {annot['Obstacle']} \
                            \nImage: {df_station['img_fpath'].iloc[i]} \
                            \nDf row: {i} \n--------------------------------")

            elif station.lower() == "slope":
                if len(annot["Car"]) == 1 and len(annot["Obstacle"]) == 9:
                    bool_empty = self._empty_annots(annotation=annot)
                    bool_missing = self._missing_annots(annotation=annot)

                    if bool_empty is True or bool_missing is True:
                        n_wrong += 1
                        list_wrong.append(i)
                        if bool_empty is True:
                            n_empty += 1
                        if bool_missing is True:
                            n_missing += 1

                        print(f"\nNo. of Cars: {len(annot['Car'])} \
                                \nNo. of Obstacles: {len(annot['Obstacle'])} \
                                \nCar Annotation: {annot['Car']} \
                                \nObstacle Annotation: {annot['Obstacle']} \
                                \nImage: {df_station['img_fpath'].iloc[i]} \
                                \nDf row: {i} \n--------------------------------")

                elif len(annot["Car"]) != 1 or len(annot["Obstacle"]) != 9:
                    n_wrong += 1
                    list_wrong.append(i)

                    bool_dup_extra, bool_dup = self._duplicate_annots(
                        station, car_annot, obstacle_annot
                    )
                    bool_empty = self._empty_annots(annotation=annot)
                    bool_missing = self._missing_annots(annotation=annot)

                    if bool_dup is True:
                        n_dups += 1

                    if bool_dup_extra is True:
                        n_dup_extra += 1

                    if bool_empty is True:
                        n_empty += 1

                    if bool_missing is True:
                        n_missing += 1

                    if len(annot["Obstacle"]) < 9:
                        n_below += 1
                        print("Number of slope obstacle annotations below default 9. \
                                \nFollow up to add additional annotations.")

                    if (bool_dup is False and bool_empty is False and \
                        bool_missing is False) and len(annot["Obstacle"]) > 9:
                        print("No empty/missing/duplicate annotations. \
                                \nManually check for extra annotations.")
                        self._extra_annots(car_annot, obstacle_annot)
                        n_extra += 1

                    print(f"\nNo. of Cars: {len(annot['Car'])} \
                            \nNo. of Obstacles: {len(annot['Obstacle'])} \
                            \nCar Annotation: {annot['Car']} \
                            \nObstacle Annotation: {annot['Obstacle']} \
                            \nImage: {df_station['img_fpath'].iloc[i]} \
                            \nDf row: {i} \n--------------------------------")

        print(f"\n############# Summary ############# \
                \nStation: {station} \
                \nTotal No. of wrong: {n_wrong} \
                \nList of wrong (df row): {list_wrong} \
                \nNo. of duplicate annotations: {n_dups} \
                \nNo. of empty annotations: {n_empty} \
                \nNo. of missing annotations (overlap with empty): {n_missing} \
                \nNo. of extra annotations: {n_extra + n_dup_extra} \
                \nNo. of slope obstacle annotations below default 9 (overlap with other categories): {n_below} \
                \n############# End #############\n\n")

        return list_wrong

    def _annot_to_df(self, car_annot: dict, obstacle_annot: dict):
        """Converts (per image) individual annotation dictionaries for
        car and obstacle respectively into dataframes.

        Args:
            car_annot (dict): car annotation
            obstacle_annot (dict): obstacle annotation

        Returns:
            tuple of pd.DataFrame: dataframes for car and obstacle
                annotation respectively
        """
        x1 = pd.DataFrame(car_annot)
        x2 = pd.DataFrame(obstacle_annot)

        return x1, x2

    def _df_to_annot(self, x1: pd.DataFrame, x2: pd.DataFrame):
        """Combines (per image) individual annotation dataframes for car and
        obstacle back to single annotation dictionary

        Args:
            x1 (pd.DataFrame): annotation dataframe for car, 
                where each row is a mask containing segmentation, area, bbox.
            x2 (pd.DataFrame): annotation dataframe for obstacle,
                where each row is a mask containing segmentation, area, bbox.

        Returns:
            new_annotation (dict): new annotation dictionary per image
        """
        for col in x1.columns:
            if col == "area":
                x1[col] = x1[col].astype("float32")
            x1[col] = x1[col].astype("object")
        for col in x2.columns:
            if col == "area":
                x2[col] = x2[col].astype("float32")
            x2[col] = x2[col].astype("object")
        new_car_annot = x1.to_dict("records")
        new_obstacle_annot = x2.to_dict("records")
        new_annotation = {
            "Car": new_car_annot,
            "Obstacle": new_obstacle_annot
        }
        return new_annotation

    def _duplicate_annots(
        self,
        station: str,
        car_annot: dict,
        obstacle_annot: dict
    ):
        """Checks for duplicate annotations in car and in obstacle annotations
        and prints image information and details if present.
        Further checks for empty/missing/extra annotations after attempting to remove duplicates
        and prints relevant information.
        Returns 2 boolean values:
            - True/False if extra annotations found/not found after removing duplicates
            - True/False if duplicates found/not found

        Args:
            station (str): accepts either "slope" or "park"
            car_annot (dict): contains car annotation
            obstacle_annot (dict): contains obstacle annotation

        Returns:
            tuple of bool: refer to docstring description
        """
        # a. Drop duplicated bump masks if any
        x1, x2 = self._annot_to_df(car_annot, obstacle_annot)
        for col in x1.columns:
            x1[col] = x1[col].astype("string")
        x1 = x1.drop_duplicates()
        for col in x2.columns:
            x2[col] = x2[col].astype("string")
        x2 = x2.drop_duplicates()

        if len(x1) < len(car_annot) or \
            len(x2) < len(obstacle_annot):
            print(f"Duplicated masks found... \
                    \nCar before: {len(car_annot)} \
                    \nCar after: {len(x1)} \
                    \nObstacle before: {len(obstacle_annot)} \
                    \nObstacle after: {len(x2)}")

            print("After removing duplicates, check for empty/missing/extra annotations...")
            new_annotation = self._df_to_annot(x1, x2)
            bool_empty = self._empty_annots(new_annotation)
            bool_missing = self._missing_annots(new_annotation)

            if bool_empty is False:
                print("No empty annotations found.")
            if bool_missing is False:
                print("No missing annotations found.")
            elif ((station == "slope") and (len(x2) < 9)):
                print("Number of slope obstacle annotations below default 9.")

            self._extra_annots(
                car_annot=new_annotation["Car"],
                obstacle_annot=new_annotation["Obstacle"]
            )
            if ((station == "slope") and (len(x1) != 1 or len(x2) != 9)) or \
                ((station == "park") and (len(x1) != 1 or len(x2) != 1)):
                print("Manually check for extra annotations.")
                return True, True   # n_dup_extra, n_dup
            else:
                print("Likely no extra annotations.")
                return False, True # n_dup_extra, n_dup
        return False, False # n_dup_extra, n_dup

    def _empty_annots(self, annotation: dict):
        """Checks for empty annotations (ie. empty list for segmentation value,
        with some value for area & bbox) and return a boolean value.

        Args:
            annotation (dict): annotation containing car and obstacle 
                segmentations, areas, bboxs

        Returns:
            bool : True if there is at least 1 empty annotation for either car or obstacle,
                False if there is completely no empty annotations for either car or obstacle.
        """
        mask_seg_car, mask_seg_obstacle = 0, []

        for k, v_list in annotation.items():
            # print(f"Key: {k}")
            # print(f"Number of masks: {len(v_list)}")
            for mask in v_list:
                # print(f"Value: {mask}")
                seg = mask["segmentation"]
                try:
                    length = len(seg[0])
                except:
                    # for empty mask segmentations with bounding box
                    length = len(seg)
                # print(f"Length of mask segmentation: {length}\n")

                if k.lower() == "car":
                    mask_seg_car = length
                elif k.lower() == "obstacle":
                    mask_seg_obstacle.append(length)

        if list(sorted(set(mask_seg_obstacle)))[0] == 0 or \
            mask_seg_car == 0:
                print(f"Empty annotations found... \
                        \nLength of car annots: {mask_seg_car} \
                        \nLength of obstacle annots: {list(mask_seg_obstacle)}")
                return True
        return False

    def _missing_annots(self, annotation: dict):
        """Checks for missing annotations (ie. no segmentation & area & bbox entry)
        and return a boolean value.

        Args:
            annotation (dict): annotation containing car and obstacle 
                segmentations, areas, bboxs

        Returns:
            bool : True if there is at least 1 missing annotation for either car or obstacle,
                False if there is completely no missing annotations for either car or obstacle.
        """
        num_missing = 0
        for k, v_list in annotation.items():
            if len(v_list) == 0:
                print(f"Missing annotation found... \
                        \nKey: {k} \nValue - No. of masks: {len(v_list)}")
                num_missing += 1

        if num_missing != 0:
            return True
        return False

    def _extra_annots(self, car_annot: dict, obstacle_annot: dict):
        """Prints (per image) obstacle annotations sorted by area for manual visual inspection
        to check for extra annotations (ie. duplicate/different segmentation/area/bbox values)
        if number of car/obstacle annotation is not 1 or 9 respectively.

        Args:
            car_annot (dict): contains car annotation
            obstacle_annot (dict): contains obstacle annotation
        """
        x1, x2 = self._annot_to_df(car_annot, obstacle_annot)
        if len(x1) != 1:
            print(f"Car Annotation: {x1.to_dict('records')} \
                    \n\nObstacle Annotation: {x2.to_dict('records')}")
            x1["area"] = x1["area"].astype("float32")
            print(x1.sort_values(by="area"))
        elif len(x2) != 9:
            print(f"Car Annotation: {x1.to_dict('records')} \
                    \n\nObstacle Annotation: {x2.to_dict('records')}")
            x2["area"] = x2["area"].astype("float32")
            print(x2.sort_values(by="area"))

    def annotation_categories(self, df: pd.DataFrame):
        """Provides value counts for annotation categories. 
        Meant for visual inspection of abnormalies, 
        assuming default is ["Car", "Obstacle"]

        Args:
            df (pd.DataFrame): input dataframe
        """
        print(f"Value counts for annotation categories: \
                \n{df['annotation_categories'].value_counts()}")
        print(df.info())

    def _extra_annotation_category(self, df: pd.DataFrame):
        """Checks if extra annotation category "Park" is used, alongside "Obstacle".
        Prints counts of various possible combinations:
        - park (p)
        - no_park (np)
        - obstacle (o)
        - no_obstacle (no)

        Args:
            df (pd.DataFrame): input dataframe
        """
        df_mpo = df[
            df["annotation_categories"] == "['Motorcycle', 'Plank', 'Obstacle']"
        ].reset_index(drop=True)

        n = len(df_mpo)
        n_park, n_obstacle = 0, 0
        park, obstacle = False, False
        p_o, p_no, np_o, np_no = 0, 0, 0, 0
        mask_seg = {}
        for i in range(0, n):
            dict1 = df_mpo["annotations"].iloc[i]
            # print(f"Park df Row: {i}")
            # print(f"Image filepath: {df_mpo['img_fpath'].iloc[i]}\n")
            for k, v_list in dict1.items():
                # print(f"Key: {k}")
                # print(f"Number of masks: {len(v_list)}")

                if k.lower() == "park":
                    if len(v_list) != 0:
                        park = True
                        n_park += 1
                    else:
                        park = False

                elif k.lower() == "obstacle":
                    if len(v_list) != 0:
                        obstacle = True
                        n_obstacle += 1
                    else:
                        obstacle = False

            if park is True and obstacle is True:
                mask_seg[i] = "p_o", df_mpo['img_fpath'].iloc[i] # add unlikely scenarios
                p_o += 1
            elif park is True and obstacle is False:
                p_no += 1
            elif park is False and obstacle is True:
                np_o += 1
            elif park is False and obstacle is False:
                mask_seg[i] = "np_no", df_mpo['img_fpath'].iloc[i] # add unlikely scenarios
                np_no += 1

        print("Park station: \nAbbrev.: park (p), no_park (np), obstacle (o), no_obstacle (no)\n")
        print(f"Num. of p_o: {p_o}, p_no: {p_no}, np_o: {np_o}, np_no: {np_no}")
        print(f"Num. of n_park: {n_park}, n_obstacle: {n_obstacle}")
        print("\nDetails (for unlikely scenarios 'p_o' and 'np_no'):")
        y = 1
        if mask_seg: 
            for k, v_tuple in mask_seg.items():
                    print(f"{y}) Park df Row: {k} \nType of annotation: {v_tuple[0]} \
                            \nImage: {v_tuple[1]}\n\n")
                    y += 1
        else:
            print("None")

    def _reverse_annotation_category(self, df: pd.DataFrame):
        """Checks if reversed annotation category ["Obstacle", "Car"] 
        is correct, based on assumption that length of obstacle segmentation 
        is always lesser than that of car.

        Args:
            df (pd.DataFrame): input dataframe
        """
        df_om = df[
            df["annotation_categories"] == "['Obstacle', 'Car']"
        ].reset_index(drop=True)

        n = len(df_om)
        n_wrong, len_seg_obstacle, len_seg_car = 0, 0, 0
        for i in range(0, n):

            annot = df_om['annotations'].iloc[i]
            for k, v_list in annot.items():
                for mask in v_list:
                    seg = mask["segmentation"]
                    # print(f"Key: {k} \nLength of Seg: {len(seg[0])}")

                # counting len of segmentation
                if k.lower() == "obstacle":
                    len_seg_obstacle = len(seg[0])
                elif k.lower() == "car":
                    len_seg_car = len(seg[0])

            # logging num of wrong annotation, assuming len(obstacle) < len(motor)
            if len_seg_obstacle > len_seg_car:
                print(f"df row: {i}, Image: {df_om['img_fpath'].iloc[i]}")
                print(f"\nLength of obstacle_seg: {len_seg_obstacle} \
                        \nLength of car_seg: {len_seg_car}")
                n_wrong += 1

        print(f"\nNum of wrong annotations: {n_wrong}")

    def plot_colour_distribution(
        self,
        df: pd.DataFrame,
        by: list,
        resize_graph: int,
        filepath: str,
        height: int = 5,
        width: int = 15
    ):
        """Generates subplots based on criteria "by", namely by:
        - station, time_of_day only, or
        - station, time_of_day, weather condition only, or
        - station, time_of_day, weather condition, pass/fail status only.
        Uses `_single_plot()` helper function.

        Args:
            df (pd.DataFrame): input dataframe
            by (list): criteria for plot, accepts only either:
                - ["station", "time_of_day"]
                - ["station", "time_of_day", "weather_condition"]
                - ["station", "time_of_day", "weather_condition", "pass_fail"]
            resize_graph (int): resize y axis from 0 to specified value, specify 0 for no resizing.
            filepath (str): filepath to save output plots in PDF format.
            height (int, optional): height of single chart in subplot. Defaults to 5.
            width (int, optional): width of single chart in subplot. Defaults to 15.
        """
        plot_dict = {}
        print(f"\n\nPlotting criteria: {by}")
        if by == ["station", "time_of_day"]:
            # get list of stations by time of day
            station_list = list(set([i for i in df["station"]]))
            station_time_list = []
            for i in station_list:
                df_station = df[df["station"] == i].reset_index(drop=True)
                time_list = list(set([i for i in df_station["time_of_day"]]))
                for j in time_list:
                    station_time = [i, j]
                    station_time_list.append(station_time)
            print("Station, Time: ", station_time_list)

            # create list of img_lists for each station
            for i in station_time_list:
                df_station = df[df["station"] == i[0]].reset_index(drop=True)
                df_station_time = df_station[df_station["time_of_day"] == i[1]].reset_index(drop=True)
                img_list = df_station_time["img_fpath"].tolist()
                plot_dict[str(i)] = img_list

        elif "weather_condition" in by:
            # get list of time of day & weather conditions for that station
            station_list = list(set([i for i in df["station"]]))

            station_weather_list = []
            for i in station_list:
                df_station = df[df["station"] == i].reset_index(drop=True)
                weather_list = list(set([i for i in df_station["weather_condition"]]))
                for j in weather_list:
                    time_list = list(
                        df.loc[
                            (df["station"] == i) & 
                            (df["weather_condition"].isin([j])),
                            "time_of_day"
                        ].unique()
                    )
                    for a in range(len(time_list)):
                        time_of_day = time_list[a]
                        station_weather = [i, time_of_day, j]
                        station_weather_list.append(station_weather)
            print("\nStation, Weather: ", station_weather_list)

            # create list of img_list for each time & weather condition
            for i in station_weather_list:
                df_station = df[df["station"] == i[0]].reset_index(drop=True)
                df_station_time = df_station[
                    df_station["time_of_day"] == i[1]
                ].reset_index(drop=True)
                df_station_time_weather = df_station_time[
                    df_station_time["weather_condition"] == i[2]
                ].reset_index(drop=True)

                # by == ["station", "time_of_day", "weather_condition"]
                if "pass_fail" not in by:
                    img_list = df_station_time_weather["img_fpath"].tolist()
                    plot_dict[str(i)] = img_list

                # by == ["station", "time_of_day", "weather_condition", "pass_fail"]
                elif "pass_fail" in by:
                    pass_fail_list = list(set(
                        [i for i in df_station_time_weather["pass_fail"]]
                    ))
                    station_weather_pass_fail_list = []
                    for j in range(len(pass_fail_list)):
                        new_list = [i[0], i[1], i[2], pass_fail_list[j]]
                        station_weather_pass_fail_list.append(new_list)
                    print("\nStation, Weather, Pass/Fail: ", station_weather_pass_fail_list)

                    # create list of img_list for each pass fail condition
                    for z in station_weather_pass_fail_list:
                        df_station = df[
                            df["station"] == z[0]
                        ].reset_index(drop=True)
                        df_station_time = df_station[
                            df_station["time_of_day"] == z[1]
                        ].reset_index(drop=True)
                        df_station_time_weather = df_station_time[
                            df_station_time["weather_condition"] == z[2]
                        ].reset_index(drop=True)
                        df_station_weather_pass_fail = df_station_time_weather[
                            df_station_time_weather["pass_fail"] == z[3]
                        ].reset_index(drop=True)
                        # create list of img_list for each pass/fail
                        img_list = df_station_weather_pass_fail["img_fpath"].tolist()
                        plot_dict[str(z)] = img_list

        # A. create fig axes based on number of keys in plot_dict
        print("\nNo. of subplots: ", len(plot_dict))
        # adjust height of fig size depending on number of features/subplots
        if len(plot_dict) > 2:
            i = round(len(plot_dict)/2)
            height = height*i

        if len(plot_dict) == 1:
            fig, axes = plt.subplots(
                nrows = 1,
                ncols = 2,
                constrained_layout = True,
                figsize = (width, height)
            )
        elif (len(plot_dict) % 2 != 0 and \
            ((len(plot_dict) - 1 ) / 2) % 2 == 0):
            fig, axes = plt.subplots(
                nrows = round(len(plot_dict)/2)+1,
                ncols = 2,
                constrained_layout = True,
                figsize = (width, height)
            )
        else:
            fig, axes = plt.subplots(
                nrows = round(len(plot_dict)/2),
                ncols = 2,
                constrained_layout = True,
                figsize = (width, height)
            )

        y = 0
        for k, v_list in plot_dict.items():
        # B. Select axes (ie. location of subplot for each key)    
            i, j = divmod(y, 2)
            # print("i and j", i, j)
            
            if len(plot_dict) == 1:
                # ax = axes
                ax = axes[i] # try this or above
            elif len(plot_dict) == 2:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            # C. plot it
            self._single_plot(
                criteria = k,
                img_list = v_list,
                resize_graph = resize_graph,
                ax = ax
            )
            y += 1

        if filepath is not None:
            pp = PdfPages(filepath)
            pp.savefig(fig, bbox_inches='tight', dpi=1000)
            pp.close()
        # plt.show()

    def _single_plot(
        self,
        criteria: str,
        img_list: list,
        resize_graph: int,
        ax: object = None
    ):
        """Generates single colour channel histogram plot from a list of image filepaths.
        To be used by `plot_colour_distribution()`.

        Args:
            criteria (str): criteria for plotting, namely the station, weather condition, pass/fail status.
            img_list (list): list of image filepaths to be plotted.
            resize_graph (int): resize y axis from 0 to specified value, specify 0 for no resizing.
            ax (object, optional): axes object for subplots in `plot_colour_distribution()`. Defaults to None.
        """
        nb_bins = 256
        count_r = np.zeros(nb_bins)
        count_g = np.zeros(nb_bins)
        count_b = np.zeros(nb_bins)

        for img_fpath in tqdm(img_list):
            image = cv2.imread(img_fpath)
            x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # reverse channel from BGR to RGB
            hist_r = np.histogram(x[0], bins=nb_bins, range=[0, 255])
            hist_g = np.histogram(x[1], bins=nb_bins, range=[0, 255])
            hist_b = np.histogram(x[2], bins=nb_bins, range=[0, 255])
            count_r += hist_r[0]
            count_g += hist_g[0]
            count_b += hist_b[0]

        bins = hist_r[1]
        ax.bar(bins[:-1], count_r, color="r", alpha=0.5, label="red")
        ax.bar(bins[:-1], count_g, color="g", alpha=0.5, label="green")
        ax.bar(bins[:-1], count_b, color="b", alpha=0.5, label="blue")
        ax.legend()
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Count")
        if resize_graph != 0:
            ax.set_ylim(0, resize_graph)
            ax.set_title(f"{criteria} - ylim: {resize_graph}", fontsize=13)
        elif resize_graph == 0:
            ax.set_title(f"{criteria} - Original", fontsize=13)

    def _convert_annotation_category(self, df: pd.DataFrame):
        """Converts `annotation_categories` string in list to nonstring in list.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            df (pd.DataFrame): output dataframe
        """
        df["annotation_categories"] = df["annotation_categories"].apply(
            ast.literal_eval
        )
        return df

def main(args):
    """Runs check for cleaned data using clean_img_data.pkl file.
    """
    # configurations
    groupby_list = [    # groupby various categories for generating counts
        ["station", "pass_fail"],
        ["station", "time_of_day", "pass_fail"],
        ["station", "weather_condition", "pass_fail"],
        ["station", "weather_condition", "time_of_day", "pass_fail"],
        ["station", "height", "width", "pass_fail"],
        ["station", "annotation_categories", "pass_fail"],
        ["station", "weather_condition", "annotation_categories", "pass_fail"]
    ]

    print("############# CHECK DATA REPORT #############")
    df = pd.read_pickle(args.pickle_path)
    print(df.transpose())
    print(df.info())

    check = Check(df = df)
    if args.with_img_channels:
        df = check.get_img_channels(df=df)
        groupby_list.append(
            ["station", "height", "width", "img_channels", "pass_fail"]
        )
    check.counts(df=df, groupby_list=groupby_list)

    list_wrong_park = check.annotation_counts(df=df, station="park")
    list_wrong_slope = check.annotation_counts(df=df, station="slope")

    check.annotation_categories(df=df)
    check._extra_annotation_category(df=df)
    check._reverse_annotation_category(df=df)

    # apply conditional if using occlusion label
    if args.occlude:
        check.occlusion_label(df=df)

    check.high_level_counts(df=df)  # for each station - num of images, cars, obstacles

    if args.with_plots:
        # multiple plots - by station, time of day
        check.plot_colour_distribution(
            df = df,
            by = ["station", "time_of_day"],
            resize_graph = 0,   # resize_graph = 500000
            filepath = args.output_plot_filepath.replace(
                "plot", "plot_station_time"
            )
        )

        # multiple plots - by station, time of day, weather condition
        check.plot_colour_distribution(
            df = df,
            by = ["station", "time_of_day", "weather_condition"],
            resize_graph = 0,   # resize_graph = 250000
            filepath = args.output_plot_filepath.replace(
                "plot", "plot_station_time_weather"
            )
        )

        # multiple plots - by station, time of day, weather condition, pass/fail
        check.plot_colour_distribution(
            df = df,
            by = ["station", "time_of_day", "weather_condition", "pass_fail"],
            resize_graph = 0,    # resize_graph = 80000
            filepath = args.output_plot_filepath.replace(
                "plot", "plot_station_time_weather_outcome"
            )
        )

    print("############# END OF REPORT #############")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check data script')
    parser.add_argument(
        '--pickle_path', type=str,
        help='Clean pickle file path. By default, refers to the one generated in Azure mount point.'
    )
    parser.add_argument(
        '--output_plot_filepath', type=str,
        help='Output file path for colour channel histogram plots. \
            By default, refers to the Azure mount point folder. \
            Filename should contain "plot.pdf" where "plot" will be replaced \
            with corresponding text for various plots. \
            Hence, do not use the string "plot" anywhere else in filepath.'
    )
    parser.add_argument(
        '--with_img_channels', default=False, dest='with_img_channels', action='store_true',
        help='This checks and generates image colour channels when specified. \
        Not necessary if all images have been verified to contain only 3 colour channels.'
    )
    parser.add_argument(
        '--with_plots', default=False, dest='with_plots', action='store_true',
        help='This generates image colour histogram plots when specified.'
    )
    parser.add_argument(
        '--occlude', default=False, dest='occlude', action='store_true',
        help='This performs occlusion label checks when specified.'
    )
    args = parser.parse_args()
    print("Args: ", args)
    
    main(args)