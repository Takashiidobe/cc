static double take9(double a, double b, double c, double d, double e, double f,
                    double g, double h, double i) {
  return a + b + c + d + e + f + g + h + i;
}

int main(void) {
  double x = 5.5;
  if ((int)x != 5) {
    return 1;
  }

  if ((unsigned int)x != 5u) {
    return 2;
  }

  unsigned long big = 1UL << 63;
  double big_d = (double)big;
  if ((unsigned long)big_d != big) {
    return 3;
  }

  long neg = -42;
  double neg_d = (double)neg;
  if ((long)neg_d != neg) {
    return 4;
  }

  double sum = take9(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
  if (sum != 45.0) {
    return 5;
  }

  double stack = take9(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5);
  if (stack != 40.5) {
    return 6;
  }

  double from_uint = (double)4000000000u;
  if ((unsigned int)from_uint != 4000000000u) {
    return 7;
  }

  return 0;
}
