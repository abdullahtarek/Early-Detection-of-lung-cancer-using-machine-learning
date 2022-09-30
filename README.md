# Early-Detection-of-lung-cancer-using-machine-learning

This project is aimed to find nodules in a 3d lung CT-scan and give each nodule a malignancy score between 0 to 4. 0 being not malignant at all and 4 being the most malignant. It used the luna dataset with 2 annatations. The <a href="https://luna16.grand-challenge.org/" >luna Dataset </a> annotations and the <a href="https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI" > LIDC IDRI </a> for the malignancy annotations. The problem had to be devided into two parts because this is a needle in a haystack problem it is not just a simple classification problem. 

<h2> Pipline </h2>
<img src="https://github.com/abdullahtarek/Early-Detection-of-lung-cancer-using-machine-learning/blob/master/Screenshot_2.png">
</br>

<h2> Skewed Data </h2>
Incremental training was used in the Googlent model because the Data was skewed. The negatives was way more than the postives so for example I trained on 100 postives and a hundered negatives and then inrementally adding more and more negatives so that the model will not always predict a negative.

<h2> Using LIDC IDRI anotations </h2>
The LUNA data set is a subset from the LIDC dataset but no previous implementation used that. so using the id in the xml annotations it was found that the same IDs were used. The xml annotations had many features for each nodule but what is used in this project was the malignancy. The nodule positions was refrenced by an edge map and the centroid of the nodule was calculated to generate the training data.


<h2> GUI </h2>
<img src="https://github.com/abdullahtarek/Early-Detection-of-lung-cancer-using-machine-learning/blob/master/Screenshot_1.png"> 
