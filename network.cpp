#include "network.h"

#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits>

/*
 * TODO: parallel for loop
 */
double4d create_d4d(int height,
                    int width,
                    int channels,
                    int numbers,
                    bool random)
{
    srand(time(NULL));
    double4d new_fmap;
    new_fmap.resize(numbers);
    for (unsigned n = 0; n < new_fmap.size(); n++) {
        new_fmap[n].resize(channels);
        for (unsigned c = 0; c < new_fmap[n].size(); c++) {
            new_fmap[n][c].resize(width);
            for (unsigned i = 0; i < new_fmap[n][c].size(); i++) {
                new_fmap[n][c][i].resize(height);
                for (unsigned j = 0; j < new_fmap[n][c][i].size(); j++)
                    new_fmap[n][c][i][j] =
                        random ? (double) (rand() % 256) : 0.0;
            }
        }
    }
    return new_fmap;
}

void init_filters(map<string, double4d> *filters, string *f_names, int2d sizes)
{
    for (unsigned i = 0; i < sizes.size(); i++) {
        double4d tmp = create_d4d(sizes[i][0], sizes[i][1], sizes[i][2],
                                  sizes[i][3], true);
        filters->insert(pair<string, double4d>(f_names[i], tmp));
    }
}

void init_biases(map<string, vector<double> > *biases,
                 string *b_names,
                 int2d sizes)
{
    srand(time(NULL));
    for (unsigned i = 0; i < sizes.size(); i++) {
        vector<double> tmp(sizes[i][3]);
        for (int j = 0; j < sizes[i][3]; j++)
            tmp[j] = rand() % 10;
        biases->insert(pair<string, vector<double> >(b_names[i], tmp));
    }
}

/*
 * TODO: parallel for loop
 */
double4d conv_layer(double4d input,
                    string layer_name,
                    map<string, double4d> filters,
                    map<string, vector<double> > biases)
{
    // zero-padding and padding-size = 1
    int padding_size = 1;
    double4d padded_input = create_d4d(input[0][0].size() + 2 * padding_size,
                                       input[0][0][0].size() + 2 * padding_size,
                                       input[0].size(), input.size(), false);

    for (unsigned n = 0; n < padded_input.size(); n++) {
        for (unsigned c = 0; c < padded_input[n].size(); c++) {
            for (unsigned i = 1; i < padded_input[n][c].size() - 1; i++) {
                for (unsigned j = 1; j < padded_input[n][c][i].size() - 1; j++)
                    padded_input[n][c][i][j] =
                        input[n][c][i - padding_size][j - padding_size];
            }
        }
    }

    int U = 1;  // stride
    double4d filter = get_filter(filters, layer_name);
    vector<double> bias = get_bias(biases, layer_name);
    /* output's (height, width, channel, numbers)
     *        = (input_height, input_width, filter_numbers, input_numbers);
     */
    double4d output = create_d4d(input[0][0].size(), input[0][0][0].size(),
                                 filter.size(), input.size(), false);
    printf("%s\t\t(%lu, %lu, %lu, %lu)\t\t%lu\t\t", layer_name.c_str(),
           output.size(), output[0][0].size(), output[0][0][0].size(),
           output[0].size(),
           filter[0][0].size() * filter[0][0][0].size() * filter[0].size() *
                   filter.size() +
               bias.size());
    int macs = 0;
    for (unsigned n = 0; n < output.size(); n++) {
        for (unsigned m = 0; m < output[0].size(); m++) {
            for (unsigned x = 0; x < output[0][0].size(); x++) {
                for (unsigned y = 0; y < output[0][0][0].size(); y++) {
                    output[n][m][x][y] = bias[m];
                    for (unsigned k = 0; k < filter[m].size();
                         k++) {  // filter_channel
                        for (unsigned i = 0; i < filter[m][k].size();
                             i++) {  // filter_height
                            for (unsigned j = 0; j < filter[m][k][i].size();
                                 j++) {  // filter_width
                                output[n][m][x][y] +=
                                    (padded_input[n][k][U * x + i][U * y + j] *
                                     filter[m][k][i][j]);
                                macs++;
                            }
                        }
                    }
                    // Activation function (ReLU)
                    output[n][m][x][y] =
                        (output[n][m][x][y] > 0) ? output[n][m][x][y] : 0.0;
                }
            }
        }
    }
    printf("%d\n", macs);
    return output;
}

/*
 * TODO: parallel for loop
 */
double4d pool_layer(double4d input,
                    string layer_name,
                    int pool_size,
                    int stride)
{
    int size = ((int) input[0][0].size() - pool_size) / stride + 1;
    int channels = (int) input[0].size();
    int numbers = (int) input.size();
    double max;
    double4d output = create_d4d(size, size, channels, numbers, false);
    printf("%s\t(%lu, %lu, %lu, %lu)  \t\t0\n", layer_name.c_str(),
           output.size(), output[0][0].size(), output[0][0][0].size(),
           output[0].size());

    for (int n = 0; n < numbers; n++) {
        for (int m = 0; m < channels; m++) {
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++) {
                    max = numeric_limits<double>::min();
                    for (int i = 0; i < pool_size; i++) {
                        for (int j = 0; j < pool_size; j++) {
                            max = (max <
                                   input[n][m][stride * x + i][stride * y + j])
                                      ? input[n][m][stride * x + i]
                                             [stride * y + j]
                                      : max;
                        }
                    }
                    output[n][m][x][y] = max;
                }
            }
        }
    }
    return output;
}

/*
 * TODO: parallel for loop
 */
double2d flatten_2d(double4d input)
{
    double2d output;
    output.resize(input.size());  // batch size
    for (unsigned i = 0; i < output.size(); i++)
        output[i].resize(input[0].size() * input[0][0].size() *
                         input[0][0][0].size());
    for (unsigned n = 0; n < input.size(); n++) {
        for (unsigned m = 0; m < input[0].size(); m++) {
            for (unsigned x = 0; x < input[0][0].size(); x++) {
                for (unsigned y = 0; y < input[0][0][0].size(); y++) {
                    output[n][m * input[0][0].size() +
                              x * input[0][0][0].size() + y] =
                        input[n][m][x][y];
                }
            }
        }
    }
    return output;
}

/*
 * TODO: parallel for loop
 */
double2d fc_layer(double2d input, string layer_name, int size)
{
    /*
     *  n: batch size, m: feature size, k: number of neurons
     *  input:  n*m,
     *  bias:   n*k,
     *  params: m*k,
     *  output: n*k,
     */
    double2d bias, params, output;

    srand(time(NULL));
    bias.resize(input.size());  // batch size
    output.resize(input.size());
    for (unsigned i = 0; i < bias.size(); i++) {
        bias[i].resize(size);
        output[i].resize(size);
        for (unsigned j = 0; j < bias[i].size(); j++) {
            bias[i][j] = (double) (rand() % 10);
            output[i][j] = 0.0;
        }
    }
    params.resize(input[0].size());
    for (unsigned i = 0; i < params.size(); i++) {
        params[i].resize(size);
        for (unsigned j = 0; j < params[i].size(); j++)
            params[i][j] = (double) (rand() % 10);
    }
    printf("%s\t(1, 1, %lu, %lu)\t\t\t%lu\t\t", layer_name.c_str(), output.size(),
           output[0].size(), params.size() * params[0].size() + bias[0].size());
    int macs = 0;
    for (unsigned n = 0; n < output.size(); n++) {
        for (unsigned k = 0; k < output[n].size(); k++) {
            output[n][k] = bias[n][k];
            for (unsigned i = 0; i < params[k].size(); i++) {
                output[n][k] += input[n][i] * params[i][k];
                macs++;
            }
        }
    }
    printf("%d\n", macs);
    return output;
}

double4d get_filter(map<string, double4d> filters, string name)
{
    map<string, double4d>::iterator iter;
    iter = filters.find(name);
    return iter->second;
}

vector<double> get_bias(map<string, vector<double> > biases, string name)
{
    map<string, vector<double> >::iterator iter;
    iter = biases.find(name);
    return iter->second;
}