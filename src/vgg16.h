#ifndef __VGG16_H
#define __VGG16_H
#include <map>
#include <string>
#include <vector>

#include "network.h"

class VGG16
{
public:
    VGG16();
    double2d forward(double4d);

private:
    void init_filters();
    void init_biases();
    double4d get_filter(std::string);
    std::vector<double> get_bias(std::string);
    double4d add_conv(double4d, std::string);
    std::map<std::string, double4d> filters;
    std::map<std::string, std::vector<double> > biases;
    std::string layer_names[13] = {"conv1_1", "conv1_2", "conv2_1", "conv2_2",
                                   "conv3_1", "conv3_2", "conv3_3", "conv4_1",
                                   "conv4_2", "conv4_3", "conv5_1", "conv5_2",
                                   "conv5_3"};
    int2d sizes{{3, 3, 3, 64},    {3, 3, 64, 64},   {3, 3, 64, 128},
                {3, 3, 128, 128}, {3, 3, 128, 256}, {3, 3, 256, 256},
                {3, 3, 256, 256}, {3, 3, 256, 512}, {3, 3, 512, 512},
                {3, 3, 512, 512}, {3, 3, 512, 512}, {3, 3, 512, 512},
                {3, 3, 512, 512}};
};
#endif