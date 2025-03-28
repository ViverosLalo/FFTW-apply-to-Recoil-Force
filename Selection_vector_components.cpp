#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <tuple>

using namespace std;

//  Función para calcular las componentes \vb{I}
double I_componentes(int l, int m, int mp) {
    if (mp == (m - 1)) {
        if ((m == 1) && (mp == 0)){
            return 0.;
        } else {
            return -1.*sqrt(double(l*(l + 1) - m*(m - 1))) / 4.0;
        }
    } else if (m == mp) {
        return  -1.*m/2. ;
    } else if (mp == (m + 1)) {
        if (m == 0){
            return -1.*sqrt(double(l*(l + 1))) / 2.0;
        } else {
            return -1.*sqrt(double(l*(l + 1) - m*(m + 1))) / 4.0;
        }
    } else {
        return 0.0; // Caso en que no cumple ninguna condición
    }
}

//  Función para calcular las componentes de \vb{Pi}
double Pi_componentes(int l, int m, int mp) {
    if (mp == (m - 1)) {
        if ((m == 1) && (mp == 0)){
            return -1.*double(l*l-1)*sqrt(double((l + 1)*l)/double((2*l + 1)*(2*l - 1))) / 2.;
        } else {
            return -1.*double(l*l-1)*sqrt(double((l + m)*(l + m - 1))/double((2*l + 1)*(2*l - 1))) / 4.;
        }
    } else if (m == mp) {
        if (m == 0){
            return double(l*l-1)*sqrt(double(l*l)/double((2*l + 1)*(2*l - 1)));
        } else {
            return double(l*l-1)*sqrt(double((l - m)*(l + m))/double((2*l + 1)*(2*l - 1))) / 2. ;
        }
    } else if (mp == (m + 1)) {
        if (m == 0){
            return double(l*l-1)*sqrt(double((l)*(l - 1))/double((2*l + 1)*(2*l - 1))) / 2.;
        } else {
            return double(l*l-1)*sqrt(double((l - m)*(l - m - 1))/double((2*l + 1)*(2*l - 1))) / 4.;
        }
    } else {
        return 0.0; // Caso en que no cumple ninguna condición
    }
}


int main() {
    int l;
    std::cout << "Introduce el valor de l: ";
    std::cin >> l;

    auto start = chrono::high_resolution_clock::now();

    // Matrices para almacenar los resultados
    std::vector<std::vector<double>> I_matriz(l + 1, std::vector<double>(l + 1));
    std::vector<std::vector<double>> PI_matriz(l + 1, std::vector<double>(l + 1));

    // Bucle para calcular los valores y llenar las matrices
    for (int m = 0; m <= l; ++m) {
        for (int mp = 0; mp <= l; ++mp) {
            I_matriz[m][mp] = I_componentes(l, m, mp);
        }
    }

    for (int m = 0; m <= l; ++m) {
        for (int mp = 0; mp < l; ++mp) {
            PI_matriz[m][mp] = Pi_componentes(l, m, mp);
        }
    }

    auto stop = chrono::high_resolution_clock::now();   //  Para el cronómetro de ejecución de cálculos
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start)/pow(10.,6.);

    //

    std::ofstream file("resultados.txt");
    if (file.is_open()) {
        file << "I_vec:\n";
        for (int m = 0; m <= l; ++m) {
            for (int mp = 0; mp <= l; ++mp) {
                file << "I[" << l << "][" << m << "][" << mp << "] = " << I_matriz[m][mp] << " ";
            }
            file << "\n";
        }

        file << "\nPi_vec:\n";
        for (int m = 0; m <= l; ++m) {
            for (int mp = 0; mp < l; ++mp) {
                file << "PI[" << l << "][" << m << "][" << mp << "] = " << PI_matriz[m][mp] << " ";
            }
            file << "\n";
        }
        file.close();
        std::cout << "Resultados guardados en resultados.txt" << std::endl;
    } else {
        std::cerr << "Error al abrir el archivo." << std::endl;
    }

    cout << "Tiempo de cómputo: " << duration.count() << " segundos" << endl;

    return 0;
}