#ifndef __NETWORK_H
#define __NETWORK_H
#include <map>
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
double2d flatten_2d(double4d);
double2d fc_layer(double2d, string, int);

double4d get_filter(map<string, double4d>, string);
vector<double> get_bias(map<string, vector<double> >, string);


#endif