
//---------------------------------------------------------------------------------------------------------
//
//          Este código calcula la componentes de la fuerza de retroceso considerando hasta el multipolo l_max=30
//          
//          
//
//          Para compilar:
//              g++ -o Aproximacion_dipolar Aproximacion_dipolar.cpp -lfftw3 -lm -lcomplex_bessel -fopenmp
//          Para ejecutar:
//              ./Aproximacion_dipolar
//
//---------------------------------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <cmath>                                        //  Para operaciones matemáticas simples
#include <vector>                                       //  Para generar y manejar arreglos (vectores)
#include <complex>                                      //  Para manejar números complejos
#include <chrono>                                       //  Para cronometrar el tiempo de cómputo
#include <fftw3.h>                                      //  Para calcular la FFTW
#include <algorithm>                                    //  Para hacer el corrimiento pre y post FFTW
#include <boost/math/special_functions/bessel.hpp>      //  Para la función modificada de Bessel del segundo tipo
#include <complex_bessel_bits/sph_besselFunctions.h>    //  Para las funciones esféricas de Bessel y Hankel
#include <omp.h>                                        //  Para paralelizar cálculos (generación de tablas)
#include "IN31.h"                                       //  Coeficientes I^{l,m}_{j,l-j} ya calculados por J. Castrejón

using namespace boost::math;
using namespace sp_bessel;
using namespace std;

// ---------- CONSTANTES ----------

double Pi           = boost::math::constants::pi<double>();
double hbar         = 4.135667731/(2*Pi) ;                  //  Constante de Plank reducida en eV*fs
double cluz         = 299.792458;                           //  Rapidez de la luz en nm/fs
double hbarcluz     = hbar*cluz;                            //  Constante de Plank reducida por la rapidez de la luz en eV*nm
double hbaromega_p  = 13.14;                                //  Frecuencia de plasma por la constante de Planck en eV
double hbargamma    = 0.197;                                //  Dammping por la constante de Planck en eV
double alphafina    = 1/137.035999139;                      //  Constante de estructura fina

//---------- FUNCIONES AUXILIARES Y MATERIALES ----------

//  Factor de Lorentz
double gammaLorentz(double beta){
    return 1. / sqrt(1. - pow(beta,2.));
}
//  Función dieléctrica del modelo de Drude
complex<double> epsilon_Drude(double hbaromega){
    return 1. - pow(hbaromega_p,2.) / (hbaromega * (hbaromega + 1i*hbargamma));
}
//  Índice de refracción del modelo de Drude
complex<double> N_Drude(double hbaromega){
    return complex<double>(sqrt(epsilon_Drude(hbaromega)));
}
//  Parámetro de tamaño
double x_tam(double hbaromega, double a){
    return hbaromega*a/hbarcluz;
}
//  Unidad imaginaria de un índice
complex<double> i_m(int m){
    if (m == 0 || m % 2 == 0) {
        return 1i; // 1i para m = 0 o m par
    } else {
        return 1; // 1 para m impar
    }
}


//---------- DEFINICIÓN DE LAS FUNCIONES DE RICCATI-BESSEL ----------

//  Función psi_l
complex<double> RB_psi(int l, complex<double> z){
    return z*sph_besselJ(l,z);
}
//  Función xi_l
complex<double> RB_xi(int l, complex<double> z){
    return z*1i*sph_hankelH1(l,z);
}
//  Derivada de la función psi_l
complex<double> D_RB_psi(int l, complex<double> z){
    return z*(double(l)*sph_besselJ(l-1,z)/double(2*l+1)-double(l+1)*sph_besselJ(l+1,z)/double(2*l+1))+sph_besselJ(l,z);
}
//  Derivada de la función xi_l
complex<double> D_RB_xi(int l, complex<double> z){
    return z*(double(l)*1i*sph_hankelH1(l-1,z)/double(2*l+1)-double(l+1)*1i*sph_hankelH1(l+1,z)/double(2*l+1))+1i*sph_hankelH1(l,z);
}


//---------- DEFINICIÓN DE LOS COEFICIENTES DE MIE ----------

//  Coeficiente de multipolos eléctricos
complex<double> tlE(int l, double hbaromega, double a){
    return -1.*(RB_psi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-N_Drude(hbaromega)*RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_psi(l,x_tam(hbaromega,a)))/
    (RB_xi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-N_Drude(hbaromega)*RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_xi(l,x_tam(hbaromega,a)));
}
//  Coeficiente de multipolos magnéticos
complex<double> tlM(int l, double hbaromega, double a){
    return -1.*(N_Drude(hbaromega)*RB_psi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_psi(l,x_tam(hbaromega,a)))/
    (N_Drude(hbaromega)*RB_xi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_xi(l,x_tam(hbaromega,a)));
}


//----------    DEFINICIÓN DE LAS FUNCIONES ESPECTRALES AUXILIARES SIMPLIFICADAS   ----------

//  Función espectral auxiliar simplificada eléctrica
complex<double> FEASelectrica(int l, int m, double a, double b, double beta, double hbaromega, double r){
    if(hbaromega>=0){
        return i_m(m)*tlE(l,hbaromega,a)*boost::math::cyl_bessel_k(m,abs(hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(1i*hbaromega*r/hbarcluz);
    } 
    else{
        return conj(i_m(m)*tlE(l,-1.*hbaromega,a)*boost::math::cyl_bessel_k(m,abs(-1.*hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(-1.*1i*hbaromega*r/hbarcluz));
    }
}
//  Función espectral auxiliar simplificada magnética
complex<double> FEASmagnetica(int l, int m, double a, double b, double beta, double hbaromega, double r){
    if(hbaromega>=0){
        return i_m(m)*tlM(l,hbaromega,a)*boost::math::cyl_bessel_k(m,abs(hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(1i*hbaromega*r/hbarcluz);
    } 
    else{
        return conj(i_m(m)*tlM(l,-1.*hbaromega,a)*boost::math::cyl_bessel_k(m,abs(-1.*hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(-1.*1i*hbaromega*r/hbarcluz));
    }
}

//----------    DEFINICION DE LOS COEFICIENTES V_{l,m}^{(*)} y W_{l,m}^{(*)}    ----------

//  Funciones factorial y doble factorial

double factorial(double numero){
    if (numero == 0.0 || numero == 1.0){
        return 1.0;
    } else if ((fmod(numero,1.0) == 0.0) && (numero > 1.0)){
        return numero*factorial(numero - 1);
    } else {
        return tgamma(numero + 1);
    }
}

double factorial2(int numero){
    if (numero == 0 || numero == 1){
        return 1.0;
    } else if (numero > 1){
        return numero*factorial2(numero - 2);
    } else {
        return pow(-1.0,(numero-1.)/2.0)*numero/factorial2(abs(numero));
    }
}

//  Coeficiente (renormalizado para obtener la constante de estructura fina) de los armónicos esféricos vectoriales

double fbarlm(int l, int m){
    return sqrt(double(2*l + 1)*factorial(l - m)/factorial(l + m));
}

complex<double> unidadim(int j, int m) {
    if ((m % 2) == 0) {
        switch (j % 4) {
            case 0: return 1.0;
            case 1: return -1i;
            case 2: return -1.0;
            case 3: return 1i;
        }
    } else {
        switch (j % 4) {
            case 0: return -1i;
            case 1: return -1.0;
            case 2: return 1i;
            case 3: return 1.0;
        }
    }
    return 0.0;
}

//  Coeficientes reales A_{l,m}^{(*)} y B_{l,m}^{(*)}
double Alm_ast(int l, int m, double beta){  //Checar qué pasa cuando m=-1
    if (m > l){
        return 0.0;
    } else {
        complex<double>  res = 0.;
        if (m >= 0){
            for (int j = m; j <= l; j++){
	            res += unidadim(j,m)*III[l-1][m][j]/(pow(2.*gammaLorentz(beta),j)*factorial(l - j)*factorial((j - m)/2.0)*factorial((j + m)/2.0));
	        }
            return fbarlm(l,m)*factorial2(2*l + 1)*res.real();
        } else {
            return pow(-1.,abs(m))*Alm_ast(l,abs(m),beta);
        }
    }
}

double Blm_ast(int l, int m, double beta){
    return Alm_ast(l, m + 1, beta)*sqrt(double((l+m+1)*(l-m))) - Alm_ast(l, m - 1, beta)*sqrt(double((l - m + 1)*(l + m)));
}

//  Coeficientes reales V_{l,m}^{(*)} y W_{l,m}^{(*)}

double Vlm_ast(int l, int m, double beta){
    return Blm_ast(l,m,beta)/(2*l*(l+1)*pow(beta,l+1)*gammaLorentz(beta));
}

double Wlm_ast(int l, int m, double beta){
    return m*Alm_ast(l,m,beta)/(l*(l+1)*pow(beta,l));
}


//----------    DEFINICIÓN DE LAS COMPONENTES DE LOS VECTORES DE SELECCIÓN  ----------

//  Función para calcular las componentes \vb{I}: interacciones homopolares
double I_comp(int l, int m, int mp) {
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

//  Función para calcular las componentes de \vb{Pi}: interacciones heteropolares
double Pi_comp(int l, int m, int mp) {
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

    inicializar_III();

    int l_max;            //  Orden de la función espectral auxiliar simplificada
    double beta;          //  Parámetros variables del problema
    int contador = 0;

    // Obtener los parámetros numéricos para la FFTW
    cout << "Introduce orden multipolar máximo: ";
    cin >> l_max;

    cout << "Introduce el valor de la rapidez relativa del electrón beta: ";
    cin >> beta;

    auto start = chrono::high_resolution_clock::now();  //  Inicia el cronómetro de ejecución de cálculos

     // Declaración de las FEAS como vectores de vectores en ambos domninios 
    vector<vector<vector<double>>> Valores_de_prueba(l_max + 1); 

    // Inicialización de los vectores para cada valor de l y m.  NOTA: resulta conveniente realizar una "extensión" de arreglos para m=-1 y m=l+1 para no tener problemas en la suma de interacciones
    for (int l = 1; l <= l_max; l++) {
        Valores_de_prueba[l].resize(l + 1);
        for (int m = 0; m <= l; m++) {
            Valores_de_prueba[l][m].resize(4, 0.0); // Inicialización con ceros
            contador += 1;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int l = 1; l <= l_max; l++){
        for (int m = 0; m <= l; m++){
            Valores_de_prueba[l][m][0] = Alm_ast(l,m,beta);
            Valores_de_prueba[l][m][1] = Blm_ast(l,m,beta);
            Valores_de_prueba[l][m][2] = Vlm_ast(l,m,beta);
            Valores_de_prueba[l][m][3] = Wlm_ast(l,m,beta);
        }
    }

    // Guardar los resultados en un archivo .txt
    ofstream file("coeficientes_ABVW.txt");
    for (int l = 1; l <= l_max; l++) {
        for (int m = 0; m <= l; m++){
            file << "A*" << "[" << l << "]" << "[" << m << "]=" << Valores_de_prueba[l][m][0] << "\t" 
                    "B*" << "[" << l << "]" << "[" << m << "]=" << Valores_de_prueba[l][m][1] << "\t" 
                    "V*" << "[" << l << "]" << "[" << m << "]=" << Valores_de_prueba[l][m][2] << "\t"
                    "W*" << "[" << l << "]" << "[" << m << "]=" << Valores_de_prueba[l][m][3] << "\n";
        }
    }
    file.close();

    auto stop = chrono::high_resolution_clock::now();   //  Para el cronómetro de ejecución de cálculos
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start)/pow(10.,6.);

    cout << "Transformada inversa de Fourier completada y resultados guardados en recoil_force_multipolar_expansion_results.txt" << endl;
    cout << "Tiempo de cómputo: " << duration.count() << " segundos" << endl;
    cout << "Se generaron " << contador << " transformadas de fourier" << endl;

    return 0;
}