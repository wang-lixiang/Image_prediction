整体代码的结构是：
Image_preprocessing-->Load_Dataset-->Data_Loader-->Load_model-->train

在Load_Dataset中可以修改数据集指定文件夹，以此来训练自己的数据集；
在Load_model中可以修改整体的模型框架，本代码使用的是resnet类的模型，读者可以自行修改；
train中包含了训练的日志，读者可以自行利用这些数据进行可视化；
checkpoint中保存了模型训练最好的一个epoch中的参数