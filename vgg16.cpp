#include <cstdio>

#include "vgg16.h"

using namespace std;

VGG16::VGG16()
{
    init_filters();
    init_biases();
}

void VGG16::init_filters()
{
    for (unsigned i = 0; i < this->sizes.size(); i++) {
        double4d tmp = create_d4d(this->sizes[i][0], this->sizes[i][1],
                                  this->sizes[i][2], this->sizes[i][3], true);
        (this->filters)
            .insert(pair<string, double4d>(this->layer_names[i], tmp));
    }
}

void VGG16::init_biases()
{
    srand(time(NULL));
    for (unsigned i = 0; i < this->sizes.size(); i++) {
        vector<double> tmp(this->sizes[i][3]);
        for (int j = 0; j < this->sizes[i][3]; j++)
            tmp[j] = rand() % 10;
        (this->biases)
            .insert(pair<string, vector<double> >(this->layer_names[i], tmp));
    }
}

double4d VGG16::get_filter(string name)
{
    map<string, double4d>::iterator iter;
    iter = this->filters.find(name);
    return iter->second;
}

vector<double> VGG16::get_bias(string name)
{
    map<string, vector<double> >::iterator iter;
    iter = this->biases.find(name);
    return iter->second;
}

double4d VGG16::add_conv(double4d input, string name)
{
    return conv_layer(input, name, get_filter(name), get_bias(name));
}

double2d VGG16::forward(double4d input)
{
    double4d conv_output;
    double2d vgg_output;
    printf("Layer\t\tMemory size\t\t\tParam #\t\tMAC #\n");
    printf(
        "======================================================================"
        "=====\n");
    conv_output = add_conv(input, "conv1_1");
    conv_output = add_conv(conv_output, "conv1_2");
    conv_output = pool_layer(conv_output, "max_pooling_1", 2, 2);

    conv_output = add_conv(conv_output, "conv2_1");
    conv_output = add_conv(conv_output, "conv2_2");
    conv_output = pool_layer(conv_output, "max_pooling_2", 2, 2);

    conv_output = add_conv(conv_output, "conv3_1");
    conv_output = add_conv(conv_output, "conv3_2");
    conv_output = add_conv(conv_output, "conv3_3");
    conv_output = pool_layer(conv_output, "max_pooling_3", 2, 2);

    conv_output = add_conv(conv_output, "conv4_1");
    conv_output = add_conv(conv_output, "conv4_2");
    conv_output = add_conv(conv_output, "conv4_3");
    conv_output = pool_layer(conv_output, "max_pooling_4", 2, 2);

    conv_output = add_conv(conv_output, "conv5_1");
    conv_output = add_conv(conv_output, "conv5_2");
    conv_output = add_conv(conv_output, "conv5_3");
    conv_output = pool_layer(conv_output, "max_pooling_5", 2, 2);
    vgg_output = flatten_2d(conv_output);
    vgg_output = fc_layer(vgg_output, "fc1_4096", 4096);
    vgg_output = fc_layer(vgg_output, "fc2_4096", 4096);
    vgg_output = fc_layer(vgg_output, "fc3_1000", 1000);
    printf(
        "======================================================================"
        "=====\n");
    return vgg_output;
}