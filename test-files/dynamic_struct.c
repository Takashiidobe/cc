void *malloc(unsigned long size);

struct Point {
  int x;
  int y;
};

int main(void) {
  struct Point *p = malloc(sizeof(struct Point));
  p->x = 1;
  p->y = 2;
  return p->x + p->y;
}
