# MagNet

## Dataset Hosting Services

- [AWS Public Dataset Program](https://aws.amazon.com/opendata/public-datasets/)
- [Google Public Data](https://www.google.com/publicdata/admin)
- [CKAN](https://ckan.org/)

## Noteworthy Datasets

- [MNIST](http://yann.lecun.com/exdb/mnist/)

> The data is stored in a very simple file format designed for storing vectors and multidimensional matrices. General info on this format is given at the end of this page, but you don't need to read that to use the data files.
>
> All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header.

- [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

> The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with cPickle.

> Loaded in this way, each of the batch files contains a dictionary with the following elements:
>  - data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
>  - labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
>
> The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
>  - label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

- [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)
