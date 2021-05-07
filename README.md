# Cocotools

## introduce
A python lib with shell entry for managing json dataset files in coco format, support detection and keypoint working

## install
- `python3 -m pip install cocotools`
- `coco --install-completion` (using this for better shell completion)

## uninstall
- `python3 -m pip uninstall cocotools`

## usage

### create coco file
1. create empty coco file from image directory
   ```bash
    coco from-image-dir IMAGE_DIR [--with-box] [-o OUTPUT_FILE]
   ```
    
2. create coco file from human labeling result
   ```bash
    coco convert-box-labeling LABEL_FILE IMAGE_DIR [-o OUTPUT_FILE]
   ```
   The labeling file should be like this:
   
   ```json
   {
       "img_name_1.jpg": {
           "data": [
               {
                   "bbox": [
                       2.598000000000013,
                       97.862,
                       152.422,
                       155.886
                   ],
                   "type": "car",
                   "values": {},
                   "id": 1
               },
               {
                   "bbox": [
                       176.67099999999996,
                       114.31700000000001,
                       160.217,
                       129.905
                   ],
                   "type": "car",
                   "values": {},
                   "id": 2
               }
           ]
       },
       "img_name_2.jpg": {
           "data": [
               {
                   "bbox": [
                       0,
                       508.394,
                       335.324,
                       497.577
                   ],
                   "type": "person",
                   "values": {},
                   "id": 1
               }
           ]
       }
   }
   ```
### visualize
Visualize box (and keypoints) result on detection (or ground truth) files
```bash
coco visualize COCO_FILE IMG_DIR
```

### print stats
Show how many images/boxes/categories in a coco file
```bash
coco print-stat COCO_FILES...
```

### evaluation
```bash
coco evaluate [OPTIONS] GT_FILE DT_FILE
```

### merge different coco files
```bash
coco merge [-o, --output FILE] INPUTS_COCOS...
```

### convert id type
1. using string id
    ```bash
    coco to-str-id COCO_FILE
    ```
1. using integer id
    ```bash
    coco to-num-id COCO_FILE
    ```

### split dataset
split dataset into train (80%) and val (20%)
```bash
coco split-dataset COCO_FILE IMAGE_DIR
```

### others
1. `coco cmd "SOME PYTHON CODE"` using python code for more operations
1. `python -m coco.badcase GT_COCO DT_COCO -i IMAGE_DIR` for visualizing badcase
//1. `python -m coco.distribution_analyze ref coco` for obtain channels which boxes similar with ref's
1. `from coco import COCO` using lib in python code

## changelog
- 0.2.0.1
    using pypi for distribution
- 0.2.0.0
    add badcase module for badcase box visualize
    add distribution_analyze module for analyze box distribution in one dataset
- 0.1.1.5
    add keep_images option for filters
- 0.1.1.4
    add debug argument
- 0.1.1.3
    support detection file with num ids
- 0.1.1.2
    add -e for evaluate
- 0.1.1.1
     fix console entry
- 0.1.1
    add argparse
- 0.1.0
    initialize the repo
