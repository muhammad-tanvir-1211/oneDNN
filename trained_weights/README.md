# Trained Weights

## About

This folder contains weights used by the Resnet50 and VGG16 examples.

See ```examples/models/README.md``` for instructions on how to use the library with these weights.

## Prerequisites

The script ```get_weights.py``` in this repo requires the following Python packages:

* numpy
* h5py

They can be installed with the following command (assuming pip installed):
```
$ python3 -m pip install --user numpy h5py
```

## Usage

To obtain the model weights simply run passing either ```vgg16``` or ```resnet50``` to download and convert the desired weights. You can also
pass ```all``` to download both sets of weights.

```
$ cd trained_weights
$ python3 get_weights.py <vgg16|resnet50|all>
```

The weights are output to ```${CWD}/<vgg16|resnet50>_transposed_param_files/```. 

**Please note**, In order to be correctly picked up by the library's CMake you **must** run the script from the ```trained_weights``` directory to ensure the weights are in the correct location.

## License

* The ResNet50 weights are ported from the ones released by [Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
* The VGG16 weights are ported from the ones [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).
