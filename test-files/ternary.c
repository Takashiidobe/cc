int main(void) {
    int x = 0;
    int y = x ? ++x : 5;
    int z = x ? 10 : (x = 3);
    int w = (x > 2) ? x : 0;
    return x + y + z + w;
}
