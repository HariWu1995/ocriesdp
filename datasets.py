import os
import io
import json

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from tqdm import tqdm
from glob import glob
from ast import literal_eval as structurize_str
from PIL import Image, ImageDraw, ImageFont

import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def load_dataset(data_folder: str or Path, dataset_type: str, **kwargs):
    if dataset_type.lower() == 'bkainaver':
        if str(data_folder)[-1].isdigit():
            kwargs['version'] = int(str(data_folder)[-1])
        dataset = load_dataset_BKAINaver(data_folder, **kwargs)
    if dataset_type.lower() == 'eaten':
        dataset = load_dataset_EATEN(data_folder, **kwargs)
    if dataset_type.lower() == 'funsd':
        dataset = load_dataset_FUNSD(data_folder, **kwargs)
    if dataset_type.lower() == 'cord':
        dataset = load_dataset_CORD(data_folder, **kwargs)
    if dataset_type.lower() == 'mcocr':
        dataset = load_dataset_MCOCR(data_folder, **kwargs)
    if dataset_type.lower() == 'sroie':
        dataset = load_dataset_SROIE(data_folder, **kwargs)

    return dataset


def gather_statistics_on_image(dataset: pd.DataFrame):
    dataset['width'], dataset['height'] = None, None
    for ri, img_path in tqdm(dataset['image_path'].iteritems(), total=len(dataset)):
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            dataset.loc[ri, ['width', 'height']] = img.size
    dataset.dropna(subset=['width', 'height'], inplace=True)
    dataset['area'] = dataset['width'] * dataset['height']

    img_stats = {
          'area' : { 'min': dataset[  'area'].min(), 'max': dataset[  'area'].max(), 'mean': dataset[  'area'].mean().astype(int), },
         'width' : { 'min': dataset[ 'width'].min(), 'max': dataset[ 'width'].max(), 'mean': dataset[ 'width'].mean().astype(int), },
        'height' : { 'min': dataset['height'].min(), 'max': dataset['height'].max(), 'mean': dataset['height'].mean().astype(int), },
    }
    return img_stats, dataset


def gather_statistics_on_bbox(dataset: pd.DataFrame, dataset_img: pd.DataFrame):
    dataset = dataset.merge(dataset_img[['image_id', 'width', 'height']], validate='m:1', 
                                     on=['image_id',]).rename(columns={f'{col}': f'{col}_image' for col in ['width', 'height']})
    dataset['text'     ] = dataset['text'].astype(str)
    dataset['text_size'] = dataset['text'].apply(len)
    dataset['LEFT_abs'], dataset['RIGHT_abs'] = None, None
    dataset['TOP_abs'], dataset['BOTTOM_abs'] = None, None
    for ri, coords in tqdm(dataset['polygons'].iteritems(), total=len(dataset)):
        coords = np.array(coords)
        dataset.loc[ri, ['LEFT_abs', 'RIGHT_abs',]] = [coords[:,0].min(), coords[:,0].max(),]
        dataset.loc[ri, ['TOP_abs', 'BOTTOM_abs',]] = [coords[:,1].min(), coords[:,1].max(),]
    dataset[['LEFT_abs', 'TOP_abs']] = dataset[['LEFT_abs', 'TOP_abs']].clip(lower=0, upper=None)

    # Absolute value
    dataset[ 'width_abs'] = (dataset[ 'RIGHT_abs'] - dataset[  'LEFT_abs'])
    dataset['height_abs'] = (dataset['BOTTOM_abs'] - dataset[   'TOP_abs'])
    dataset[  'area_abs'] =  dataset[ 'width_abs'] * dataset['height_abs']
    bbox_abs = {
          'area' : { 'min':  dataset[  'area_abs'].min(), 'max': dataset[  'area_abs'].max(), 'mean': dataset[  'area_abs'].mean().astype(int), },
         'width' : { 'min':  dataset[ 'width_abs'].min(), 'max': dataset[ 'width_abs'].max(), 'mean': dataset[ 'width_abs'].mean().astype(int), },
        'height' : { 'min':  dataset['height_abs'].min(), 'max': dataset['height_abs'].max(), 'mean': dataset['height_abs'].mean().astype(int), },
           'top' :           dataset[   'TOP_abs'].min(),
        'bootom' :           dataset['BOTTOM_abs'].max(),
          'left' :           dataset[  'LEFT_abs'].min(),
         'right' :           dataset[ 'RIGHT_abs'].max(),
    }

    # Relative value to width / height
    dataset[ 'width_rel'] = (dataset[ 'RIGHT_abs'] - dataset[  'LEFT_abs']) /  dataset[ 'width_image']
    dataset['height_rel'] = (dataset['BOTTOM_abs'] - dataset[   'TOP_abs']) /  dataset['height_image']
    dataset[  'area_rel'] = (dataset[ 'width_abs'] * dataset['height_abs']) / (dataset['height_image'] * dataset['width_image'])
    dataset[   'TOP_rel'] =  dataset[   'TOP_abs']                          /  dataset['height_image']
    dataset['BOTTOM_rel'] =  dataset['BOTTOM_abs']                          /  dataset['height_image']
    dataset[  'LEFT_rel'] =  dataset[  'LEFT_abs']                          /  dataset[ 'width_image']
    dataset[ 'RIGHT_rel'] =  dataset[ 'RIGHT_abs']                          /  dataset[ 'width_image']
    bbox_rel = {
          'area' : { 'min': dataset[  'area_rel'].min(), 'max': dataset[  'area_rel'].max(), 'mean': dataset[  'area_rel'].mean(), },
         'width' : { 'min': dataset[ 'width_rel'].min(), 'max': dataset[ 'width_rel'].max(), 'mean': dataset[ 'width_rel'].mean(), },
        'height' : { 'min': dataset['height_rel'].min(), 'max': dataset['height_rel'].max(), 'mean': dataset['height_rel'].mean(), },
           'top' :          dataset[   'TOP_rel'].min(),
        'bootom' :          dataset['BOTTOM_rel'].max(),
          'left' :          dataset[  'LEFT_rel'].min(),
         'right' :          dataset[ 'RIGHT_rel'].max(),
    }

    bbox_count = dataset['image_id'].value_counts()
    bbox_stats = {
        'count' : { 'min':  bbox_count.min(), 
                    'max':  bbox_count.max(), 
                   'mean':  bbox_count.mean().astype(int), },
        'absolute' : bbox_abs,
        'relative' : bbox_rel,
    }
    return bbox_stats, dataset


def gather_statistics(dataset: dict):

    # Statistics
    img_stats, dataset['image'] = gather_statistics_on_image(dataset['image'])
    bbox_stats, dataset['label'] = gather_statistics_on_bbox(dataset['label'], dataset['image'])

    char_list = ''.join(dataset['label']['text'].astype(str).values.tolist())
    char_stats = dict(Counter(char_list))
    
    general_stats = {
        "num_chars"  : len(char_stats),
        "num_images" : len(dataset['image']),
    }
    
    dataset['stats'] = {
         "meta" : general_stats,
        "image" : img_stats,
         "bbox" : bbox_stats,
         "text" : { 'count' : { 'min' : dataset['label']['text_size'].min(), 
                                'max' : dataset['label']['text_size'].max(),
                               'mean' : dataset['label']['text_size'].mean().astype(int), },
                    'vocab' : char_stats, }
    }
    return dataset


def infer_statistics(stats: str or Path or dict) -> dict:
    if isinstance(stats, (str, Path)):
        with open(stats, 'r') as f_reader:
            stats = json.load(f_reader)
    if not isinstance(stats, (dict,)):
        raise TypeError(f"input must be 1 of str, pathlib.Path or dict while yours are {stats.__class__}")
    
    return {
        'image_size'  : max(stats['image'][ 'width']['max'], 
                            stats['image']['height']['max'],),
      'max_text_len'  :     stats[ 'text'][ 'count']['max'],
     'max_bbox_count' :     stats[ 'bbox'][ 'count']['max'],
     'max_bbox_size'  : max(stats[ 'bbox']['absolute'][ 'width']['max'], 
                            stats[ 'bbox']['absolute']['height']['max'],),
     'min_bbox_size'  : min(stats[ 'bbox']['absolute'][ 'width']['min'], 
                            stats[ 'bbox']['absolute']['height']['min'],),
              'vocab' : set(stats[ 'text'][ 'vocab'].keys()),
    }


def load_dataset_BKAINaver(data_folder: str or Path, image_extension: str = 'jpg', 
                                                     label_extension: str = 'txt', version: int = 1):
    
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    if not image_extension.startswith('.'):
        image_extension = f'.{image_extension}'
    if not label_extension.startswith('.'):
        label_extension = f'.{label_extension}'
    
    print(f'\n\n Processing BKAINaver dataset {data_folder} ...')
    image_folder = data_folder / 'train_images'
    label_folder = data_folder / 'train_labels'

    images_df = pd.DataFrame(columns=['image_id', 'image_path'])
    labels_df = pd.DataFrame(columns=['image_id', 'polygons', 'text'])
            
    for label_file in tqdm(os.listdir(label_folder)):
        if not label_file.endswith(label_extension):
            continue
        
        # Load list of paths to images
        image_name = label_file[3:-len(label_extension)]
        if version == 1:
            image_file = image_folder / f"{image_name}{image_extension}"
            image_id = int(str(image_file).split('_')[-1][:-len(image_extension)])
        elif version == 2:
            image_id = int(image_name)
            image_file = image_folder / f"im{image_id:04}{image_extension}"
        else:
            raise ValueError(f'Expecting version is either 1 or 2, while input is {version}!')
        images_df.loc[len(images_df)] = [image_id, image_file]
            
        # Load labels
        label_file = label_folder / label_file
        with open(label_file, 'r', encoding='utf-8') as f_reader:
            labels = f_reader.readlines()
        for label in labels:
            *bbox, bb_text = label.strip().split(',')[:9]
            bbox = np.array(bbox).reshape((-1,2)).astype(int).tolist()
            labels_df.loc[len(labels_df)] = [image_id, bbox, bb_text]
    
    return { 'image': images_df, 
             'label': labels_df, }


def load_dataset_EATEN(data_folder: str or Path, image_extension: str = 'jpg', 
                                                 label_extension: str = 'tsv'):
    
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    if not image_extension.startswith('.'):
        image_extension = f'.{image_extension}'
    if not label_extension.startswith('.'):
        label_extension = f'.{label_extension}'
    
    print(f'\n\n Processing EATEN dataset {data_folder} ...')
    image_folder = data_folder / 'images'
    label_folder = data_folder / 'labels'

    images_df = pd.DataFrame(columns=['image_id', 'image_path'])
    labels_df = pd.DataFrame(columns=['image_id', 'polygons', 'text', 'text_type'])

    for label_file in tqdm(os.listdir(label_folder)):
        if not label_file.endswith(label_extension):
            continue
        
        # Load list of paths to images
        image_name = label_file[:-len(label_extension)]
        image_file = image_folder / f"{image_name}{image_extension}"
        image_id = image_file.name[:-len(image_extension)]
        images_df.loc[len(images_df)] = [image_id, image_file]
            
        # Load labels
        label_file = label_folder / label_file
        with open(label_file, 'r', encoding='utf-8') as f_reader:
            labels = f_reader.readlines()
        for label in labels:
            *bbox, bb_text, bb_type = label.strip().split(',')[1:11]
            bbox = np.array(bbox).reshape((-1,2)).astype(int).tolist()
            labels_df.loc[len(labels_df)] = [image_id, bbox, bb_text, bb_type]
    
    return { 'image': images_df, 
             'label': labels_df, }


def load_dataset_CORD(data_folder: str or Path, image_extension: str = 'jpg', 
                                                label_extension: str = 'csv'):
    
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    if not image_extension.startswith('.'):
        image_extension = f'.{image_extension}'
    if not label_extension.startswith('.'):
        label_extension = f'.{label_extension}'
    
    print(f'\n\n Processing CORD dataset {data_folder} ...')

    def resize(bbox: str) -> list:
        bbox = structurize_str(bbox)
        return np.array(bbox).reshape((-1,2)).astype(int).tolist()
    
    images_df = pd.DataFrame(columns=['image_id', 'image_path'])
    labels_df = pd.DataFrame(columns=['image_id', 'category', 'group_id', 'subgroup_id',
                                      'polygons', 'key_info', 'row_id', 'text',])

    for subset in ['train', 'validation', 'test']:
        subset_dir = data_folder / subset
            
        for label_file in tqdm(os.listdir(subset_dir)):
            if not label_file.endswith(label_extension):
                continue
            if not label_file.startswith('OCR'):
                continue
                
            # Read labels for OCR
            df_ocr = pd.read_csv(str(subset_dir / label_file))
            df_ocr['polygons'] = df_ocr['polygons'].apply(lambda x: resize(x))
            df_ocr['image_id'] = subset + '_' + df_ocr['image_id'].apply(lambda x: f"{x:04}")
            labels_df = pd.concat([labels_df, df_ocr], axis=0)
            
            # Verify images
            for img_id in df_ocr['image_id'].unique():
                img_path = subset_dir / f'images/img_{img_id[-4:]}.jpg'
                if os.path.isfile(img_path):
                    images_df.loc[len(images_df)] = [img_id, img_path]
    
    return { 'image': images_df, 
             'label': labels_df, }


def load_dataset_FUNSD(data_folder: str or Path, image_extension: str = 'png', label_level: str = 'word'):
    
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    if not image_extension.startswith('.'):
        image_extension = f'.{image_extension}'

    def resize(bbox: str) -> list:
        bbox = structurize_str(bbox)
        return np.array(bbox).reshape((-1,2)).astype(int).tolist()
    
    print(f'\n\n Processing FUNSD dataset {data_folder} ...')
    labels_df = pd.read_csv(str(data_folder / f'data_{label_level}.csv'))
    labels_df['polygons'] = labels_df['polygons'].apply(resize)
    labels_df['image_id'] = labels_df['subset'] + '_' + labels_df['image_id']
    labels_df = labels_df.drop(columns=['subset'])

    images_df = pd.DataFrame(columns=['image_id', 'image_path'])
        
    for img_id in labels_df['image_id'].unique():
        subset, img_idx = img_id.split('_', 1)
        img_path = str(data_folder / f'{subset}ing_data/images/{img_idx}.png')
        if os.path.isfile(img_path):
            images_df.loc[len(images_df)] = [img_id, img_path]

    return { 'image': images_df, 
             'label': labels_df, }


def load_dataset_MCOCR(data_folder: str or Path, image_extension: str = 'jpg', 
                                                 label_extension: str = 'tsv',):
    
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    if not image_extension.startswith('.'):
        image_extension = f'.{image_extension}'
    if not label_extension.startswith('.'):
        label_extension = f'.{label_extension}'
    
    print(f'\n\n Processing MCOCR dataset {data_folder} ...')
    data_folder = data_folder / '5_key_information_extraction'
    image_folder = data_folder / 'images'
    label_folder = data_folder / 'labels'

    images_df = pd.DataFrame(columns=['image_id', 'image_path'])
    labels_df = pd.DataFrame(columns=['image_id', 'polygons', 'text', 'text_type'])
            
    for label_file in tqdm(os.listdir(label_folder)):
        if not label_file.endswith(label_extension):
            continue
        
        # Load list of paths to images
        image_name = label_file[:-len(label_extension)]
        image_file = image_folder / f"{image_name}{image_extension}"
        images_df.loc[len(images_df)] = [image_name, image_file]
            
        # Load labels
        label_file = label_folder / label_file
        with open(label_file, 'r', encoding='utf-8') as f_reader:
            labels = f_reader.readlines()
        for label in labels:
            label_list = label.strip().split(',')[1:]
            bbox    =          label_list[:8]
            bb_text = ','.join(label_list[8:-1])
            bb_type =          label_list[-1]
            # if len(label_list) > 10:
            #     print(bb_text)
            bbox = np.array(bbox).reshape((-1,2)).astype(int).tolist()
            labels_df.loc[len(labels_df)] = [image_name, bbox, bb_text, bb_type]
    
    return { 'image': images_df, 
             'label': labels_df, }


def load_dataset_SROIE(data_folder: str or Path, image_extension: str = 'jpg', 
                                                 label_extension: str = 'txt',):
    
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    if not image_extension.startswith('.'):
        image_extension = f'.{image_extension}'
    if not label_extension.startswith('.'):
        label_extension = f'.{label_extension}'
    
    print(f'\n\n Processing SROIE dataset {data_folder} ...')

    def map_entity_to_bbox(line: str, entities: pd.DataFrame):
        line_set = line.replace(",", "").strip().split()
        for i, column in enumerate(entities):
            entity_values = entities.iloc[0, i].replace(",", "").strip()
            entity_set = entity_values.split()
            
            matches_count = 0
            for l in line_set:
                if any(SequenceMatcher(a=l, b=b).ratio() > 0.8 for b in entity_set):
                    matches_count += 1
                
                if (column.upper() == 'ADDRESS' and (matches_count / len(line_set)) >= 0.5) \
                or (column.upper() != 'ADDRESS' and (matches_count == len(line_set))) \
                or matches_count == len(entity_set):
                    return column.upper()
        return "O"

    def map_all_entities(words: pd.DataFrame, entities: pd.DataFrame):
        max_area = { 
            "TOTAL": (0, -1), 
             "DATE": (0, -1),
        }  # Value, index
        already_labeled = {
              "TOTAL" : False,
               "DATE" : False,
            "ADDRESS" : False,
            "COMPANY" : False,
                  "O" : False,
        }

        # Go through every line in $words and assign it a label
        labels = []
        for i, line in enumerate(words['text']):
            label = map_entity_to_bbox(line, entities)

            already_labeled[label] = True
            if (label == "ADDRESS" and already_labeled["TOTAL"]) \
            or (label == "COMPANY" and (already_labeled["DATE"] or already_labeled["TOTAL"])):
                label = "O"

            # Assign to the largest bounding box
            if label in ["TOTAL", "DATE"]:
                bbox = words.loc[i, 'polygons']
                w_values = np.array(bbox[0::2]).astype(int)
                h_values = np.array(bbox[1::2]).astype(int)
                area = (w_values.max() - w_values.min()) + (h_values.max() - h_values.min())

                if max_area[label][0] < area:
                    max_area[label] = (area, i)

                label = "O"

            labels.append(label)

        labels[max_area[ "DATE"][1]] = "DATE"
        labels[max_area["TOTAL"][1]] = "TOTAL"

        words["text_type"] = labels
        return words

    images_df = pd.DataFrame(columns=['image_id', 'image_path'])
    labels_df = pd.DataFrame(columns=['image_id', 'polygons', 'text', 'text_type'])

    for subset in ['train', 'test']:
        subset_dir = data_folder / subset
        image_folder = subset_dir / 'img'
        lbndbox_folder = subset_dir / 'box'
        lentity_folder = subset_dir / 'entities'

        for label_file in tqdm(os.listdir(lbndbox_folder)):
            if not label_file.endswith(label_extension):
                continue
            
            # Load list of paths to images
            image_name = label_file[:-len(label_extension)]
            image_file = image_folder / f"{image_name}{image_extension}"
            image_name = image_name + f'_{subset}'
            images_df.loc[len(images_df)] = [image_name, image_file]
                
            # Load labels
            label_df = pd.DataFrame(columns=labels_df.columns)
            lbbox_file = lbndbox_folder / label_file
            with open(lbbox_file, 'r') as f_reader:
                labels = f_reader.readlines()
            for label in labels:
                label_list = label.strip().split(',')
                if len(label_list) < 9:
                    print(label_list)
                    continue
                bbox    =          label_list[:8]
                bb_text = ','.join(label_list[8:])
                bbox = np.array(bbox).reshape((-1,2)).astype(int).tolist()
                label_df.loc[len(label_df)] = [image_name, bbox, bb_text, None]

            lentty_file = lentity_folder / label_file
            with open(lentty_file, 'r') as f_reader:
                entities = pd.DataFrame([json.load(f_reader)])
            label_df = map_all_entities(label_df, entities)
            labels_df = pd.concat([labels_df, label_df], axis=0)
    
    return { 'image': images_df, 
             'label': labels_df, }


def extract_dataset_CORD(data_folder):
    """
    Download: https://huggingface.co/datasets/naver-clova-ix/cord-v2/tree/main/data
    """
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    filenames = glob(str(data_folder / '*.parquet'))

    for fn in filenames:
        
        print(f"\n\n Extracting data in {fn} ...")
        
        # Preparation
        fn_split = fn.split('-')
        subset_dir = fn_split[0]
        if not os.path.isdir(subset_dir):
            os.makedirs(subset_dir)

        fn_new = subset_dir + '/{}' + f"_{int(fn_split[1])}.csv"
        
        image_dir = Path(subset_dir) / 'images'
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        
        # Extraction
        df = pd.read_parquet(fn, engine='fastparquet')
        df_kie = pd.DataFrame(columns=['image_id', 'menu', 'total',])
        df_ocr = pd.DataFrame(columns=['image_id', 'category', 'group_id', 'subgroup_id',
                                       'polygons', 'key_info', 'row_id', 'text',])
            
        for _, (img_labels, img_data, _) in tqdm(df.iterrows(), total=len(df)):
            img_labels = structurize_str(img_labels)
            # print(img_labels.keys())
            # print(json.dumps(img_labels['valid_line'], indent=4))

            # Meta data
            img_id   =  img_labels['meta']['image_id']
            img_meta =  img_labels['meta']['image_size']
            img_size = (img_meta['width'], img_meta['height'])
            
            # Extract OCR
            for line in img_labels['valid_line']:
                words = line['words']
                for word in words:
                    coords = [word['quad'][f'{ax}{i+1}'] for i in range(4) for ax in ['x', 'y']]
                    df_ocr.loc[len(df_ocr)] = [img_id, line['category'], line['group_id'], line['sub_group_id'],
                                                coords, word['is_key'], word['row_id'], word['text']]
            
            # Extract KIE
            kie = img_labels['gt_parse']
            df_kie.loc[len(df_kie)] = [img_id, kie.get('menu'), kie.get('total'),]

            # Extract image
            img_path =  image_dir / f'img_{img_id:04}.jpg'
            img_data = Image.open(io.BytesIO(img_data))
            img_data.save(img_path, 'jpeg')
            
        # Save dataframes
        df_kie.to_csv(fn_new.format('KIE'), index=False)
        df_ocr.to_csv(fn_new.format('OCR'), index=False)


def extract_dataset_FUNSD(data_folder: str or Path, label_extension: str = 'json'):
    
    assert isinstance(data_folder, (str, Path))
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    if not label_extension.startswith('.'):
        label_extension = f'.{label_extension}'
    
    print(f'\n\n Extracting data in {data_folder} ...')
    df_phrase = pd.DataFrame(columns=['subset', 'image_id', 'phrase_id', 'polygons', 'text', 'text_type', 'connections'])
    df_word = pd.DataFrame(columns=['subset', 'image_id', 'phrase_id', 'polygons', 'text'])
        
    for subset in ['train', 'test']:
        label_folder = data_folder / f"{subset}ing_data" / 'annotations'

        for label_file in tqdm(os.listdir(label_folder)):
            if not label_file.endswith(label_extension):
                continue
            
            image_id = label_file[:-len(label_extension)]
               
            # Load labels
            with open(label_folder / label_file, 'r', encoding='utf-8') as f_reader:
                all_labels = json.load(f_reader)['form']
                for label in all_labels:
                    
                    # Extract phrase-level labels
                    phrase_text = label['text']
                    phrase_type = label['label']
                    phrase_bbox = label['box']
                    phrase_link = label['linking']
                    phrase_id = label['id']
                    df_phrase.loc[len(df_phrase)] = [subset, image_id, phrase_id, phrase_bbox, phrase_text, phrase_type, phrase_link]
                    
                    # Extract word-level labels
                    for word in label['words']:
                        df_word.loc[len(df_word)] = [subset, image_id, phrase_id, word['box'], word['text']]

    df_phrase.to_csv(str(data_folder / 'data_phrase.csv'), index=False)
    df_word.to_csv(str(data_folder / 'data_word.csv'), index=False)




