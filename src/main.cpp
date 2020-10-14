#include "utils.h"
#include "vgg16.h"

int main()
{
    double4d input = create_d4d(224, 224, 3, 1, true);
    VGG16 model = VGG16();
    model.forward(input);
    return 0;
}