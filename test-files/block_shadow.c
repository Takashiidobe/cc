int main(void) {
    int x = 1;
    {
        int x = 2;
        if (x > 1) {
            x = 3;
            int x = 4;
        }
        x = x + 10;
    }
    return x;
}
