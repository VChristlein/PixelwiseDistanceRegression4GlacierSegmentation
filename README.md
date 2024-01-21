# Pixelwise Distance Regression for Glacier Calving Front Detection and Segmentation

Code by Christoph Baller and AmirAbbas Davari. All credits to them.

## Abstract
Glacier calving front position (CFP) is an important glaciological variable.
Traditionally, delineating the CFPs has been carried out manually, which was
subjective, tedious, and expensive. Automating this process is crucial for
continuously monitoring the evolution and status of glaciers. Recently, deep
learning approaches have been investigated for this application. However, the
current methods get challenged by a severe class imbalance problem. In this
work, we propose to mitigate the class imbalance between the calving front
class and the noncalving front class by reformulating the segmentation problem
into a pixelwise regression task. A convolutional neural network (CNN) gets
optimized to predict the distance values to the glacier front for each pixel in
the image. The resulting distance map localizes the CFP and is further
postprocessed to extract the calving front line. We propose three
postprocessing methods, one method based on statistical thresholding, a second
method based on conditional random fields (CRFs), and finally the use of a
second U-Net. The experimental results confirm that our approach significantly
outperforms the state-of-the-art methods and produces accurate delineation. The
second U-Net obtains the best performance results, resulting in an average
improvement of about 21% Dice coefficient enhancement.

## Cite
```
@ARTICLE{9732953,
  author={Davari, Amirabbas and Baller, Christoph and Seehaus, Thorsten and Braun, Matthias and Maier, Andreas and Christlein, Vincent},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Pixelwise Distance Regression for Glacier Calving Front Detection and Segmentation}, 
  year={2022},
  volume={60},
  number={},
  pages={1-10},
  keywords={},
  doi={10.1109/TGRS.2022.3158591},
  ISSN={1558-0644},
  month={},}

```

code_zone: former code src directory for the first models (enze zhang).
This code base is also used for the evaluation for the later comparsion with this study.
The dataset is split into patches.


usage:
```
S0_data_info.py [-h] [--patch_size PATCH_SIZE] [--plot_model] [--load_model LOAD_MODEL] [--chained_training CHAINED_TRAINING]

Dataset Generator

optional arguments:
  -h, --help            show this help message and exit
  --patch_size PATCH_SIZE
                        Set the patch_size
  --plot_model          create model png
  --load_model LOAD_MODEL
                        load model from given path
  --chained_training CHAINED_TRAINING
                        use chained jobs for model training
```


usage: 
```
main.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-p PATCH_SIZE] [-o OUT_PATH]
               [-t TIME] [-l {cross,focal,dice}] [--optimizer {adam,sgd}]
               [--clr CLR] [--plot_model] [--load_model LOAD_MODEL]
               [--chained_training CHAINED_TRAINING]
Glacier Front Segmentation
optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        number of training epochs (integer value > 0)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (integer value)
  -p PATCH_SIZE, --patch_size PATCH_SIZE
                        patch size (integer value)
  -o OUT_PATH, --out_path OUT_PATH
                        output path for results
  -t TIME, --time TIME  timestamp for model saving
  -l {cross,focal,dice}, --loss {cross,focal,dice}
                        loss function for the deep classifiers training
                        (binary_crossentropy/f1_loss)
  --optimizer {adam,sgd}
                        optimizer for the deep classifiers training (adam/sgd)
  --clr CLR             use cyclic learning rate (0/1)
  --plot_model          create model png
  --load_model LOAD_MODEL
                        load model from given path
  --chained_training CHAINED_TRAINING
                        use chained jobs for model training
```


usage: 
```
evaluation.py [-h] [--model_path MODEL_PATH] [--patch_size PATCH_SIZE] [--out_path OUT_PATH] [--zone_model ZONE_MODEL]
```

Evaluation

```
optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to trained model
  --patch_size PATCH_SIZE
                        patch size (integer value)
  --out_path OUT_PATH   output path for results
  --zone_model ZONE_MODEL
                        evaluate zone model (0/1)
```


Example:
```
python3 S0_data_info.py --patch_size 256
python3 main.py --batch_size 8 --epochs 200 --loss cross --patch_size 256
python3 evaluation.py --model_path saved_model/20200621-120712_cross --patch_size 256
```


--------------------------------

This is the new code base for the studied approach with distance maps.
It is also used for the evaluation for later comparison with this study where no patches are generated.


```
usage: main.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-o OUT_PATH] [-t TIME] [--target_size TARGET_SIZE] [--gamma GAMMA] [--second_model SECOND_MODEL] [--loss LOSS]
```

Glacier Front Segmentation
```
optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        number of training epochs (integer value > 0)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (integer value)
  -o OUT_PATH, --out_path OUT_PATH
                        output path for results
  -t TIME, --time TIME  timestamp for model saving
  --target_size TARGET_SIZE
                        input size of images
  --gamma GAMMA         Gamma value for distance map creation
  --second_model SECOND_MODEL
                        create second model
  --loss LOSS
```
Example:
```
python3 main.py --epochs 200 --gamma 7 --second_model 0 --loss mse --batch_size 8
```


--------------------------------

Final evaluation: This steps need to be done to create the final qualitative and quantitative evaluation for both code directories.


usage: 
```
eval.py [-h] [--pred_path PRED_PATH] [--test_path TEST_PATH] [--save_path SAVE_PATH] [--method METHOD]

Evaluation

optional arguments:
  -h, --help            show this help message and exit
  --pred_path PRED_PATH
                        Path to the predicted test images
  --test_path TEST_PATH
                        Path to test images
  --save_path SAVE_PATH
                        Path to save the evaluation
  --method METHOD       Set the method that was used for prediction

'src/data_512_7/test/'
save path
```
Example:
```
python3 eval.py --pred_path src/results/qualitative/bce --test_path src/data_512_7/test --save_path src/report/bce --method bce
```

Complete Example:
```
python3 main.py --epochs 200 --gamma 7 --second_model 0 --loss mse --batch_size 8
python3 eval.py --pred_path src/report/7/qualitative/front --test_path src/data_512_7/test --save_path src/results/threshold --method threshold
```
