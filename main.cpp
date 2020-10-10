#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace std;

typedef vector<vector<int> > int2d;
typedef vector<vector<double> > double2d;
// double4d [batches or numbers][channels][height][weidth]
typedef vector<vector<double2d> > double4d;

/* Create a double4d tensor for a specific size with values depend on a bool
 * parameter. If true, fill in with random value, otherwise, zero.
 */
double4d create_d4d(int, int, int, int, bool);
// Initialize the filters for VGG16 network architecture
void init_filters(map<string, double4d> *, string *, int2d);
void init_biases(map<string, vector<double> > *, string *, int2d);
/* Do convolution with 1 stide and padding strategy, zero-padding and 'same',
 * which implied the size of the output feature map is the same as the input
 */
double4d conv_layer(double4d,
                    string,
                    map<string, double4d>,
                    map<string, vector<double> >);
double4d pool_layer(double4d, string, int, int);

/* TODO: fc_layer */


double4d get_filter(map<string, double4d>, string);
vector<double> get_bias(map<string, vector<double> >, string);

int main()
{
    double4d input = create_d4d(224, 224, 3, 1, true);
    map<string, double4d> filters;
    map<string, vector<double> > biases;
    map<string, double4d>::iterator iter;
    map<string, vector<double> >::iterator iter1;
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

    double4d output;
    printf("Layer\t\tOutput shape\t\t\tParam #\n");
    printf("============================================================\n");
    output = conv_layer(input, "conv1_1", filters, biases);
    output = conv_layer(output, "conv1_2", filters, biases);
    output = pool_layer(output, "max_pooling_1", 2, 2);

    output = conv_layer(output, "conv2_1", filters, biases);
    output = conv_layer(output, "conv2_2", filters, biases);
    output = pool_layer(output, "max_pooling_2", 2, 2);

    output = conv_layer(output, "conv3_1", filters, biases);
    output = conv_layer(output, "conv3_2", filters, biases);
    output = conv_layer(output, "conv3_3", filters, biases);
    output = pool_layer(output, "max_pooling_3", 2, 2);

    output = conv_layer(output, "conv4_1", filters, biases);
    output = conv_layer(output, "conv4_2", filters, biases);
    output = conv_layer(output, "conv4_3", filters, biases);
    output = pool_layer(output, "max_pooling_4", 2, 2);

    output = conv_layer(output, "conv5_1", filters, biases);
    output = conv_layer(output, "conv5_2", filters, biases);
    output = conv_layer(output, "conv5_3", filters, biases);
    output = pool_layer(output, "max_pooling_5", 2, 2);
    printf("============================================================\n");

    return 0;
}

double4d create_d4d(int height,
                    int width,
                    int channels,
                    int numbers,
                    bool random)
{
    srand(time(NULL));
    double4d new_fmap;
    new_fmap.resize(numbers);
    for (int n = 0; n < new_fmap.size(); n++) {
        new_fmap[n].resize(channels);
        for (int c = 0; c < new_fmap[n].size(); c++) {
            new_fmap[n][c].resize(width);
            for (int i = 0; i < new_fmap[n][c].size(); i++) {
                new_fmap[n][c][i].resize(height);
                for (int j = 0; j < new_fmap[n][c][i].size(); j++)
                    new_fmap[n][c][i][j] =
                        random ? (double) (rand() % 256) : 0.0;
            }
        }
    }
    return new_fmap;
}

void init_filters(map<string, double4d> *filters, string *f_names, int2d sizes)
{
    for (int i = 0; i < sizes.size(); i++) {
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
    for (int i = 0; i < sizes.size(); i++) {
        vector<double> tmp(sizes[i][3]);
        for (int j = 0; j < sizes[i][3]; j++)
            tmp[j] = rand() % 10;
        biases->insert(pair<string, vector<double> >(b_names[i], tmp));
    }
}

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

    for (int n = 0; n < padded_input.size(); n++) {
        for (int c = 0; c < padded_input[n].size(); c++) {
            for (int i = 1; i < padded_input[n][c].size() - 1; i++) {
                for (int j = 1; j < padded_input[n][c][i].size() - 1; j++)
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
    printf("%s\t\t(%lu, %lu, %lu, %lu)\t\t%lu\n", layer_name.c_str(),
           output.size(), output[0][0].size(), output[0][0][0].size(),
           output[0].size(),
           filter[0][0].size() * filter[0][0][0].size() * filter[0].size() *
                   filter.size() +
               bias.size());

    for (int n = 0; n < output.size(); n++) {
        for (int m = 0; m < output[0].size(); m++) {
            for (int x = 0; x < output[0][0].size(); x++) {
                for (int y = 0; y < output[0][0][0].size(); y++) {
                    output[n][m][x][y] = bias[m];
                    for (int k = 0; k < filter[m].size();
                         k++) {  // filter_channel
                        for (int i = 0; i < filter[m][k].size();
                             i++) {  // filter_height
                            for (int j = 0; j < filter[m][k][i].size();
                                 j++) {  // filter_width
                                output[n][m][x][y] +=
                                    (padded_input[n][k][U * x + i][U * y + j] *
                                     filter[m][k][i][j]);
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
    return output;
}

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