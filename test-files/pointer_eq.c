int main(void) {
  int a = 1;
  int b = 2;
  int *p = &a;
  int *q = &a;
  int *r = &b;
  return (p == q) * 5 + (p == r);
}
