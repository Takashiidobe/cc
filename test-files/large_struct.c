struct Point {
  unsigned int a;
  unsigned long b;
  unsigned short c;
  unsigned long long d;
  unsigned long long e;
  unsigned long long f;
  unsigned long long g;
  unsigned long long h;
  unsigned long long i;
};

int main(void) {
  struct Point p;
  p.a = 1;
  p.b = 2;
  p.c = 3;
  p.d = 4;
  p.e = 5;
  p.f = 6;
  p.g = 7;
  p.h = 8;
  p.i = 9;

  return p.a + p.b + p.c + p.d + p.e + p.f + p.g + p.h + p.i;
}
