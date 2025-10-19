static double scale(double x, double y) {
  return x * y + 1.0;
}

int main(void) {
  double a = 1.5;
  double b = 2.0;
  double c = a + b;
  if (c != 3.5) {
    return 1;
  }

  c -= 1.0;
  if (c != 2.5) {
    return 2;
  }

  double d = scale(c, 4.0);
  if (d != 11.0) {
    return 3;
  }

  if (scale(0.0, -1.0) != 1.0) {
    return 4;
  }

  double e = 5.0;
  ++e;
  if (e != 6.0) {
    return 5;
  }

  e--;
  if (e != 5.0) {
    return 6;
  }

  if (!(c < e)) {
    return 7;
  }

  if (c > e) {
    return 8;
  }

  return 0;
}
