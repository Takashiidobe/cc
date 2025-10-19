int main(void) {
  int x = 10;
  int *y = &x;
  int **z = &y;
  return **z;
}
