#ifndef __NETWORK_H
#define __NETWORK_H
#include <string>

#include "utils.h"

/* Do convolution with 1 stide and padding strategy, zero-padding and 'same',
 * which implied the size of the output feature map is the same as the input
 */
double4d conv_layer(double4d, std::string, double4d, std::vector<double>);
double4d pool_layer(double4d, std::string, int, int);
double2d flatten_2d(double4d);
double2d fc_layer(double2d, std::string, int);

#endif