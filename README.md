Conditional GAN
====
Conditional Generative Adversarial Networks for anime generation (AnimeGAN).

![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/GAN_result.gif) <br />
Training results dump every 500 min-batch in 25 epoch(26000th min-batch) for the following tags
- blue hair blue eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/0.jpg)
- gray hair green eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/1.jpg)
- green hair red eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/2.jpg)
- orange hair brown eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/3.jpg)
- blonde hair gray eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/4.jpg)
- pink hair aqua eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/5.jpg)

## Sample training data 

![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/train.jpg) <br />

## Environment
python3 <br />
tensorflow 1.0 <br />
scipy <br />

## Model structure

![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/model.jpg)

## Data
[source link](https://drive.google.com/open?id=0BwJmB7alR-AvMHEtczZZN0EtdzQ) <br />
[google drive link]()

## Usage 
1. Download hw3 data from data link, place the MLDS_HW3_dataset/ in the same directory and unzip the face.zip in MLDS_HW3_dataset/
2. Replace the tags in MLDS_HW3_dataset/sample_testing_text.txt to the right format. 
3. Start training !

## Train
First time use, you need to do the preprocessing
```
$ python3 main.py --prepro 1
```
If you already have done the preprocessing
```
$ python3 main.py --prepro 0
```
## Model
- dcgan structure
- use one hot encoding for condition tags

## Test 
This code will automatically dump the results for the tags specified in MLDS_HW3_dataset/sample_testing_text.txt every <em>dump_every</em> batches to the test_img/ folder. <br />

## Testing tags format
```
1,<Color> hair <Color> eyes 
2,<Color> hair <Color> eyes
3,<Color> hair <Color> eyes
4,<Color> hair <Color> eyes
.
.
.
```
- Possible colors for eyes
```
['<UNK>', 'yellow', 'gray', 'blue', 'brown', 'red', 'green', 'purple', 'orange',
 'black', 'aqua', 'pink', 'bicolored']
```
- Possible colors for hair
```
['<UNK>', 'gray', 'blue', 'brown', 'red', 'blonde', 'green', 'purple', 'orange',
 'black', 'aqua', 'pink', 'white']
```









