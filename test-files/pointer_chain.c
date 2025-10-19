int main(void) {
  int base = 3;
  int *p = &base;
  int **pp = &p;
  **pp = **pp * 4;
  return base;
}
