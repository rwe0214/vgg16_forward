#include "utils.h"

#include <ctime>

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