Conditional GAN
====
Generative Adversarial Networks for anime generation.

![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/result.gif) <br />
training result in 25 epoch for the following tags
- blue hair blue eyes
- blue hair green eyes
- green hair red eyes
- green hair pink eyes
- blue hair yellow eyes
- pink hair aqua eyes

## Environment
python3 <br />
tensorflow 1.0 <br />
scipy <br />

## Data
[source link](https://drive.google.com/open?id=0BwJmB7alR-AvMHEtczZZN0EtdzQ) <br />
[google drive link]()

## Usage 
Download hw3 data from data link<br />

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
dcgan structure

## Inference 
This code provide automatically dump the results for the tags specified in MLDS_HW3_dataset/sample_testing_text.txt every <em>dump_every<em> batches to the test_img/ folder. <br />

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









