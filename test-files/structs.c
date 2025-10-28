struct Point {
  int x;
  int y;
};

int main(void) {
  struct Point p;
  p.x = 1;
  p.y = 2;

  return p.x + p.y;
}
