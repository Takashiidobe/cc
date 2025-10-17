int main(void) {
    int x = 0;
    if (1) {
        x = 5;
    }
    if (0) {
        x = 99;
    } else {
        x += 1;
    }
    if (x > 5) {
        x += 10;
    }
    return x;
}
