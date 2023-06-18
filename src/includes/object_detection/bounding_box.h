#ifndef B_BOX
#define B_BOX

struct BoundingBox {
  float xmin, ymin, xmax, ymax;
  float confidence;
};

#endif // !B_BOX