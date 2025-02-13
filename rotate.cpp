#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Definir el arreglo
    std::vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int N = arr.size();
    int shift = N / 2 - 1; // Número de posiciones a desplazar

    // Imprimir el arreglo original
    std::cout << "Arreglo original: ";
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // Realizar el corrimiento (shift) del arreglo
    std::rotate(arr.begin(), arr.begin() + shift, arr.end());

    // Imprimir el arreglo después del corrimiento
    std::cout << "Arreglo después del corrimiento: ";
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}