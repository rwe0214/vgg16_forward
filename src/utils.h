#ifndef __UTILS_H
#define __UTILS_H
#include <map>
#include <vector>

typedef std::vector<std::vector<int> > int2d;
typedef std::vector<std::vector<double> > double2d;
// double4d [batches or numbers][channels][height][weidth]
typedef std::vector<std::vector<double2d> > double4d;

/* Create a double4d tensor for a specific size with values depend on a bool
 * parameter. If true, fill in with random value, otherwise, zero.
 */
double4d create_d4d(int, int, int, int, bool);

#endif