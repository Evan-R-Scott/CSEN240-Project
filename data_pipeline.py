import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Preprocessor:
    def __init__(self, path , categories):
        self.data_path = path
        self.categories = categories
        self.label_encoder = None
        self.target_names = None
        self.class_to_index = {cat: idx for idx, cat in enumerate(categories)}
    
    def load_data(self, path):
        image_paths = []
        labels = []

        loc = os.path.join(self.data_path, path)

        for category in self.categories:
            category_path = os.path.join(loc, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                image_paths.append(image_path)
                labels.append(category)
        
        df = pd.DataFrame({"image_path": image_paths, "label": labels})
        print(df.shape)

        return df

    def show_data(self, df):
        print(df.duplicated().sum())
        print(df.isnull().sum())
        print(df.info())
        print("Unique labels: {}".format(df['label'].unique()))
        print("Label counts: {}".format(df['label'].value_counts()))
    
    def encode_labels(self, df):
        df['category_encoded'] = df['label'].map(self.class_to_index).astype(str)
        return df[['image_path', 'category_encoded']]
        # self.label_encoder = LabelEncoder()
        # df['category_encoded'] = self.label_encoder.fit_transform(df['label'])
        # return df[['image_path', 'category_encoded']]
    
    def balance_data(self, df):
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(df[['image_path']],
        df['category_encoded'])
        df_resampled = pd.DataFrame(X_resampled, columns=['image_path'])
        df_resampled['category_encoded'] = y_resampled
        print("\nClass distribution after oversampling:")
        print(df_resampled['category_encoded'].value_counts())
        print(df_resampled)
        df_resampled['category_encoded'] = df_resampled['category_encoded'].astype(str)
        return df_resampled

    def create_generators(self, train_df, valid_df, test_df, img_size, batch_size):
        # tr_gen = ImageDataGenerator(
        #     rescale=1./255,
        #     rotation_range=20,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     shear_range=0.15,
        #     zoom_range=0.15,
        #     horizontal_flip=True,
        #     fill_mode="nearest"
        #     )
        tr_gen = ImageDataGenerator(rescale=1./255)
        ts_gen = ImageDataGenerator(rescale=1./255)
        
        train_gen_new = tr_gen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='category_encoded',
            target_size=img_size,
            class_mode='sparse',
            # class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            batch_size=batch_size,
            classes=['0', '1', '2']
        )

        valid_gen_new = ts_gen.flow_from_dataframe(
            valid_df,
            x_col='image_path',
            y_col='category_encoded',
            target_size=img_size,
            class_mode='sparse',
            # class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            batch_size=batch_size,
            classes=['0', '1', '2']
        )

        test_gen_new = ts_gen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col='category_encoded',
            target_size=img_size,
            class_mode='sparse',
            # class_mode='categorical',
            color_mode='rgb',
            shuffle=False,
            batch_size=batch_size,
            classes=['0', '1', '2']
        )

        self.target_names = test_gen_new.class_indices
        return train_gen_new, valid_gen_new, test_gen_new
    
    def preprocess(self, img_size=(224,224), batch_size=16):
        df = self.load_data("train")
        self.show_data(df)
        df = self.encode_labels(df)
        df = self.balance_data(df)
        train_df, test_df = train_test_split(
            df,
            train_size=0.9,
            shuffle=True,
            random_state=42,
            stratify=df['category_encoded']
        )
        val_df = self.load_data("val")
        val_df = self.encode_labels(val_df)
        val_df['category_encoded'] = val_df['category_encoded'].astype(str)

        train_gen, valid_gen, test_gen = self.create_generators(train_df, val_df, test_df, img_size=img_size, batch_size=batch_size)
        return train_gen, valid_gen, test_gen