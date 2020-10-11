#include <cstdio>
#include <map>
#include <string>

#include "network.h"

using namespace std;

int main()
{
    double4d input = create_d4d(224, 224, 3, 1, true);
    map<string, double4d> filters;
    map<string, vector<double> > biases;
    string layer_names[13] = {"conv1_1", "conv1_2", "conv2_1", "conv2_2",
                              "conv3_1", "conv3_2", "conv3_3", "conv4_1",
                              "conv4_2", "conv4_3", "conv5_1", "conv5_2",
                              "conv5_3"};
    // {{height, width, channel, numbers}}
    int2d sizes{{3, 3, 3, 64},    {3, 3, 64, 64},   {3, 3, 64, 128},
                {3, 3, 128, 128}, {3, 3, 128, 256}, {3, 3, 256, 256},
                {3, 3, 256, 256}, {3, 3, 256, 512}, {3, 3, 512, 512},
                {3, 3, 512, 512}, {3, 3, 512, 512}, {3, 3, 512, 512},
                {3, 3, 512, 512}};

    init_filters(&filters, layer_names, sizes);
    init_biases(&biases, layer_names, sizes);

    double4d conv_output;
    double2d vgg_output;
    printf("Layer\t\tMemory size\t\t\tParam #\t\tMAC #\n");
    printf(
        "======================================================================"
        "=====\n");
    conv_output = conv_layer(input, "conv1_1", filters, biases);
    conv_output = conv_layer(conv_output, "conv1_2", filters, biases);
    conv_output = pool_layer(conv_output, "max_pooling_1", 2, 2);

    conv_output = conv_layer(conv_output, "conv2_1", filters, biases);
    conv_output = conv_layer(conv_output, "conv2_2", filters, biases);
    conv_output = pool_layer(conv_output, "max_pooling_2", 2, 2);

    conv_output = conv_layer(conv_output, "conv3_1", filters, biases);
    conv_output = conv_layer(conv_output, "conv3_2", filters, biases);
    conv_output = conv_layer(conv_output, "conv3_3", filters, biases);
    conv_output = pool_layer(conv_output, "max_pooling_3", 2, 2);

    conv_output = conv_layer(conv_output, "conv4_1", filters, biases);
    conv_output = conv_layer(conv_output, "conv4_2", filters, biases);
    conv_output = conv_layer(conv_output, "conv4_3", filters, biases);
    conv_output = pool_layer(conv_output, "max_pooling_4", 2, 2);

    conv_output = conv_layer(conv_output, "conv5_1", filters, biases);
    conv_output = conv_layer(conv_output, "conv5_2", filters, biases);
    conv_output = conv_layer(conv_output, "conv5_3", filters, biases);
    conv_output = pool_layer(conv_output, "max_pooling_5", 2, 2);
    vgg_output = flatten_2d(conv_output);
    vgg_output = fc_layer(vgg_output, "fc1_4096", 4096);
    vgg_output = fc_layer(vgg_output, "fc2_4096", 4096);
    vgg_output = fc_layer(vgg_output, "fc3_1000", 1000);
    printf(
        "======================================================================"
        "=====\n");

    return 0;
}