long diff(int *a, int *b) { return a - b; }

int main(void) {
  int seq[6];
  seq[0] = 1;
  seq[1] = 1;
  seq[2] = 2;
  seq[3] = 3;
  seq[4] = 5;
  seq[5] = 8;

  int *hi = seq + 5;
  int *lo = seq + 2;
  return diff(hi, lo);
}
