void putchar(int c);

void print_long_inner(long v) {
  if (v <= -10 || v >= 10) {
    print_long_inner(v / 10);
  }
  {
    int d = (int)(v % 10);
    if (d < 0) {
      d = -d;
    }
    putchar(48 + d);
  }
}

void print_long(long v) {
  if (v < 0) {
    putchar(45);
  }
  print_long_inner(v);
}

int main(void) {
  unsigned int num = ~0U;

  unsigned long ull = (unsigned long)num + 3;

  print_long(ull);

  return ull;
}
