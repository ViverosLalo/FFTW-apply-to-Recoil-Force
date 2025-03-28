
//---------------------------------------------------------------------------------------------------------
//
//          Este código calcula la componentes de la fuerza de retroceso considerando hasta el multipolo l_max=30
//          
//          
//
//          Para compilar:
//              g++ -o Recoil_force_multipole_expansion Recoil_force_multipole_expansion.cpp IN31.cpp -lfftw3 -lm -lcomplex_bessel -fopenmp
//          Para ejecutar:
//              ./Recoil_force_multipole_expansion
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

double Blm_ast(int l, int m, double beta){  //Checar qué pasa cuando m=0
    return Alm_ast(l,m+1,beta)*sqrt(double((l+m+1)*(l-m)))-Alm_ast(l,m-1,beta)*sqrt(double((l-m+1)*(l+m)));
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

    inicializar_III();      // Inicializa la lista de Coeficientes I^{l,m}_{j,l-j} calculada por J. Castrejón

    int l_max;                  //  Orden de la función espectral auxiliar simplificada
    double a, b, beta;          //  Parámetros variables del problema
    const double r = cluz/2;    //  Radio de integración de la superficie: se escoge c/2 para que el pulso generado se centre en t=0.5 fs
    int contador=0;

    int n;                              //  Potencia de 2 para generar el número de datos
    double T;                           //  El dominio temporal será [-T,T]
    const double tolerancia = 1e-12;    //  Umbral para reemplazar valores pequeños por cero
    double t_min, t_max;                //  Intervalo de tiempo para acotar el número de datos en Mathematica y pueda procesarlos

    char opcion;
    char resultado;

    do {// Obtener los parámetros numéricos para la FFTW
        cout << "Introduce la potencia de 2 para el número de datos (entero positivo): ";
        cin >> n;
        cout << "Introduce el fin del intervalo T: ";
        cin >> T;

        int NN = static_cast<int>(pow(2.,n));    //  Se genera el número de datos como una cantidad entera 

        // Calcular la cantidad
        double delta_omega= Pi / T;
        double omega_inf = (-(NN/2)+1)*delta_omega;
        double omega_sup = (-(NN/2)+NN)*delta_omega;
        cout << "El dominio espectral será [" << hbar*omega_inf << "," << hbar*omega_sup << "] eV" << " con pasos de " << hbar*delta_omega << " eV" << endl;

        // Preguntar al usuario si desea continuar o reiniciar
        cout << "¿Quieres ejecutar el resto del programa (e) o volver a introducir n y T (r)? (e/r): ";
        cin >> opcion;

        // Validar la entrada del usuario
        while (opcion != 'e' && opcion != 'r') {
            std::cout << "Opción inválida. Por favor elige 'e' para ejecutar o 'r' para reiniciar: ";
            std::cin >> opcion;
        }

    } while (opcion == 'r');


    /* Obtener los parámetros numéricos para la FFTW
    cout << "Introduce la potencia de 2 para el número de datos (entero positivo): ";
    cin >> n;
    cout << "Introduce orden multipolar máximo: ";
    cin >> l_max;
    cout << "Introduce el fin del intervalo T: ";
    cin >> T;*/

    // Obtener los parámetros físicos del sistema
    cout << "Introduce orden multipolar máximo: ";
    cin >> l_max;
    cout << "Introduce el valor del radio a [nm]: ";
    cin >> a;
    cout << "Introduce el valor del parámetro de impacto b [nm]: ";
    cin >> b;
    cout << "Introduce el valor de la rapidez relativa del electrón beta (0<beta<1): ";
    cin >> beta;
    cout << "Introduce el valor mínimo del intervalo de tiempo [fs]: ";
    cin >> t_min;
    cout << "Introduce el valor máximo del intervalo de tiempo [fs]: ";
    cin >> t_max;
    //cout << "¿Quieres calcular todas las contribuciones hasta l_max (t) o sólo contribución de l_max (s)? (t/s)";
    //cin >> resultado;


    auto start = chrono::high_resolution_clock::now();  //  Inicia el cronómetro de ejecución de cálculos

    int N = static_cast<int>(pow(2.,n));

    // Declaración de las FEAS como vectores de vectores en ambos domninios 
    vector<vector<vector<complex<double>>>> OmegaE_matriz_in_freq_domain(l_max + 1); 
    vector<vector<vector<complex<double>>>> OmegaM_matriz_in_freq_domain(l_max + 1);
    vector<vector<vector<complex<double>>>> OmegaE_matriz_in_time_domain(l_max + 1);
    vector<vector<vector<complex<double>>>> OmegaM_matriz_in_time_domain(l_max + 1);

    //if (resultado == 't'){
    // Inicialización de los vectores para cada valor de l y m.
    for (int l = 1; l <= l_max; l++) {
        OmegaE_matriz_in_freq_domain[l].resize(l + 2);
        OmegaM_matriz_in_freq_domain[l].resize(l + 2);
        OmegaE_matriz_in_time_domain[l].resize(l + 2);
        OmegaM_matriz_in_time_domain[l].resize(l + 2);
        for (int m = 0; m <= l; m++) {
            OmegaE_matriz_in_freq_domain[l][m].resize(N, 0.0); // Inicialización con ceros
            OmegaM_matriz_in_freq_domain[l][m].resize(N, 0.0);
            OmegaE_matriz_in_time_domain[l][m].resize(N, 0.0);
            OmegaM_matriz_in_time_domain[l][m].resize(N, 0.0);
        }
    }

    double delta_omega= Pi / T;
    for(int l = 1; l <= l_max; l++){
        for(int m = 0; m <= l; m++){
            //----------    Calcular los valores de las FEAS en frecuencias
            #pragma omp parallel for
            for(int j = 0; j < N; j++) {
                double omega = (-(N/2)+1+ j)*delta_omega + numeric_limits<double>::epsilon(); //Se suma una épsilon de máquina para evitar divergencias en omega=0
                OmegaE_matriz_in_freq_domain[l][m][j] = FEASelectrica(l, m, a, b, beta, hbar*omega, r);
                if (m!=0){
                    OmegaM_matriz_in_freq_domain[l][m][j] = FEASmagnetica(l, m, a, b, beta, hbar*omega, r);
                }
            }
    
            //---------- Realizar el corrimiento de los arreglos para la FFT ----------
            int shift = (N/2)-1;
            rotate(OmegaE_matriz_in_freq_domain[l][m].begin(), OmegaE_matriz_in_freq_domain[l][m].begin() + shift, OmegaE_matriz_in_freq_domain[l][m].end());
            rotate(OmegaM_matriz_in_freq_domain[l][m].begin(), OmegaM_matriz_in_freq_domain[l][m].begin() + shift, OmegaM_matriz_in_freq_domain[l][m].end());

            // Declarar plan de FFTW para transformada inversa compleja a "compleja"
            fftw_plan pE = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE_matriz_in_freq_domain[l][m].data()), reinterpret_cast<fftw_complex*>(OmegaE_matriz_in_time_domain[l][m].data()), FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_plan pM = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaM_matriz_in_freq_domain[l][m].data()), reinterpret_cast<fftw_complex*>(OmegaM_matriz_in_time_domain[l][m].data()), FFTW_FORWARD, FFTW_ESTIMATE);

            // Ejecutar plan de FFTW
            fftw_execute(pE);
            fftw_execute(pM);

            if (m==0){
                contador += 1;
            } else {
                contador += 2;
            }

            //Revertir el corrimiento (shift) del arreglo
            rotate(OmegaE_matriz_in_time_domain[l][m].begin(), OmegaE_matriz_in_time_domain[l][m].begin() + (N - shift), OmegaE_matriz_in_time_domain[l][m].end());
            rotate(OmegaM_matriz_in_time_domain[l][m].begin(), OmegaM_matriz_in_time_domain[l][m].begin() + (N - shift), OmegaM_matriz_in_time_domain[l][m].end());

            // Escalar los resultados
            #pragma omp parallel for
            for (int j = 0; j < N; j++) {
                OmegaE_matriz_in_time_domain[l][m][j] *= delta_omega/(2*Pi); // Factor de escala dado por la partición en el dominio espectral y convención de transformada
                if (abs(OmegaE_matriz_in_time_domain[l][m][j].real()) <= tolerancia) OmegaE_matriz_in_time_domain[l][m][j].real(0.0);
                OmegaM_matriz_in_time_domain[l][m][j] *= delta_omega/(2*Pi); // Factor de escala dado por la partición en el dominio espectral y convención de transformada
                if (abs(OmegaM_matriz_in_time_domain[l][m][j].real()) <= tolerancia) OmegaM_matriz_in_time_domain[l][m][j].real(0.0);
            }
            fftw_destroy_plan(pE);
            fftw_destroy_plan(pM);
            //cout << "se crearon las FEAS eléctrica y magnética de orden [" << l << "][" << m << "]" << endl;
       }
    }

    //cout << "Error al calcular las contribuciones a la fuerza de retroceso" << endl;


    // Calcular las componentes de la fuerza de retroceso en aproximación dipolar
    vector<double> Fuerza_de_retroceso_homo_X(N);
    vector<double> Fuerza_de_retroceso_hetero_X(N);
    vector<double> Fuerza_de_retroceso_hetero_Elec_X(N);
    vector<double> Fuerza_de_retroceso_hetero_Magn_X(N);
    vector<double> Fuerza_de_retroceso_total_X(N);
    vector<double> Fuerza_de_retroceso_homo_Z(N);
    vector<double> Fuerza_de_retroceso_hetero_Z(N);
    vector<double> Fuerza_de_retroceso_hetero_Elec_Z(N);
    vector<double> Fuerza_de_retroceso_hetero_Magn_Z(N);
    vector<double> Fuerza_de_retroceso_total_Z(N);
    
    #pragma omp parallel for /*(se acorta el tiempo a la mitad pero la workstation empieza a sonar mucho)*/
    for (int j = 0; j < N ; j++){
        for (int l = 1; l <= l_max; l++){
            for (int m = 0; m <= l; m++){
                if (m == 0){
                    Fuerza_de_retroceso_homo_X[j] += 4.0*(-1.0*Vlm_ast(l,0,beta)*Wlm_ast(l,1,beta)*OmegaE_matriz_in_time_domain[l][0][j].real()*OmegaM_matriz_in_time_domain[l][1][j].real() + 
                                                               Vlm_ast(l,1,beta)*Wlm_ast(l,0,beta)*OmegaE_matriz_in_time_domain[l][1][j].real()*OmegaM_matriz_in_time_domain[l][0][j].real())*I_comp(l,0,1);
                } else if (m == l){
                    Fuerza_de_retroceso_homo_X[j] += 4.0*pow(-1.0,m)*(-1.0*Vlm_ast(l,l,beta)*Wlm_ast(l,l-1,beta)*OmegaE_matriz_in_time_domain[l][l][j].real() * OmegaM_matriz_in_time_domain[l][l-1][j].real() + 
                                                                           Vlm_ast(l,l-1,beta)*Wlm_ast(l,l,beta)*OmegaE_matriz_in_time_domain[l][l-1][j].real()*OmegaM_matriz_in_time_domain[l][l][j].real()) * I_comp(l,l,l-1);
                } else {
                    Fuerza_de_retroceso_homo_X[j] += 4.0*pow(-1.0,m)*((-1.0*Vlm_ast(l,m,beta)*Wlm_ast(l,m-1,beta)*OmegaE_matriz_in_time_domain[l][m][j].real()*   OmegaM_matriz_in_time_domain[l][m-1][j].real() + 
                                                                            Vlm_ast(l,m-1,beta)*Wlm_ast(l,m,beta)*OmegaE_matriz_in_time_domain[l][m-1][j].real()* OmegaM_matriz_in_time_domain[l][m][j].real())*I_comp(l,m,m-1) +
                                                                      (-1.0*Vlm_ast(l,m,beta)*Wlm_ast(l,m+1,beta)*OmegaE_matriz_in_time_domain[l][m][j].real()*   OmegaM_matriz_in_time_domain[l][m+1][j].real() + 
                                                                            Vlm_ast(l,m+1,beta)*Wlm_ast(l,m,beta)*OmegaE_matriz_in_time_domain[l][m+1][j].real()* OmegaM_matriz_in_time_domain[l][m][j].real())*I_comp(l,m,m+1));  //Componente transversal fuerza homopolar
                }
                if (l == 1){
                    Fuerza_de_retroceso_hetero_X[j] += 0.0;
                    Fuerza_de_retroceso_hetero_Elec_X[j] += 0.0;
                    Fuerza_de_retroceso_hetero_Magn_X[j] += 0.0;
                } else {
                    if (m == 0){
                        Fuerza_de_retroceso_hetero_Elec_X[j] += -8.0*(Vlm_ast(l,0,beta)*Vlm_ast(l-1,1,beta)*OmegaE_matriz_in_time_domain[l][0][j].real()*OmegaE_matriz_in_time_domain[l-1][1][j].real())*Pi_comp(l,0,1);
                        Fuerza_de_retroceso_hetero_Magn_X[j] += +8.0*(Wlm_ast(l,0,beta)*Wlm_ast(l-1,1,beta)*OmegaM_matriz_in_time_domain[l][0][j].real()*OmegaM_matriz_in_time_domain[l-1][1][j].real())*Pi_comp(l,0,1);
                        Fuerza_de_retroceso_hetero_X[j] += Fuerza_de_retroceso_hetero_Elec_X[j] + Fuerza_de_retroceso_hetero_Magn_X[j];
                    } else if (m >= (l-1)){
                        Fuerza_de_retroceso_hetero_Elec_X[j] += -8.0*(Vlm_ast(l,m,beta)*Vlm_ast(l-1,l-1,beta)*OmegaE_matriz_in_time_domain[l][m][j].real()*OmegaE_matriz_in_time_domain[l-1][m-1][j].real())*Pi_comp(l,l,l-1);
                        Fuerza_de_retroceso_hetero_Magn_X[j] += +8.0*(Wlm_ast(l,m,beta)*Wlm_ast(l-1,l-1,beta)*OmegaM_matriz_in_time_domain[l][m][j].real()*OmegaM_matriz_in_time_domain[l-1][m-1][j].real())*Pi_comp(l,l,l-1);
                        Fuerza_de_retroceso_hetero_X[j] += Fuerza_de_retroceso_hetero_Elec_X[j] + Fuerza_de_retroceso_hetero_Magn_X[j];
                    } else {
                        Fuerza_de_retroceso_hetero_Elec_X[j] += -8.0*((Vlm_ast(l,m,beta)*Vlm_ast(l-1,m-1,beta)*OmegaE_matriz_in_time_domain[l][m][j].real()*OmegaE_matriz_in_time_domain[l-1][m-1][j].real())*Pi_comp(l,m,m-1) +
                                                                      (Vlm_ast(l,m,beta)*Vlm_ast(l-1,m+1,beta)*OmegaE_matriz_in_time_domain[l][m][j].real()*OmegaE_matriz_in_time_domain[l-1][m+1][j].real())*Pi_comp(l,m,m+1));  //Componente transversal fuerza heteropolar
                        Fuerza_de_retroceso_hetero_Magn_X[j] += +8.0*((Wlm_ast(l,m,beta)*Wlm_ast(l-1,m-1,beta)*OmegaM_matriz_in_time_domain[l][m][j].real()*OmegaM_matriz_in_time_domain[l-1][m-1][j].real())*Pi_comp(l,m,m-1) +
                                                                      (Wlm_ast(l,m,beta)*Wlm_ast(l-1,m+1,beta)*OmegaM_matriz_in_time_domain[l][m][j].real()*OmegaM_matriz_in_time_domain[l-1][m+1][j].real())*Pi_comp(l,m,m+1));
                        Fuerza_de_retroceso_hetero_X[j] += Fuerza_de_retroceso_hetero_Elec_X[j] + Fuerza_de_retroceso_hetero_Magn_X[j];
                    }
                }
                Fuerza_de_retroceso_homo_Z[j] += 8.0*pow(-1.0,m)*Vlm_ast(l,m,beta)*Wlm_ast(l,m,beta)*OmegaE_matriz_in_time_domain[l][m][j].real()*OmegaM_matriz_in_time_domain[l][m][j].real()*I_comp(l,m,m);                                                                                                                                                    //Componente transversal fuerza homopolar
                if ((l == 1) || (m > (l-1))){
                    Fuerza_de_retroceso_hetero_Z[j] += 0.0;
                    Fuerza_de_retroceso_hetero_Elec_Z[j] += 0.0;
                    Fuerza_de_retroceso_hetero_Magn_Z[j] += 0.0;
                } else {
                    Fuerza_de_retroceso_hetero_Elec_Z[j] += -8.0*(Vlm_ast(l,m,beta)*Vlm_ast(l-1,m,beta)*OmegaE_matriz_in_time_domain[l][m][j].real()*OmegaE_matriz_in_time_domain[l-1][m][j].real())*Pi_comp(l,m,m);              //Componente transversal fuerza heteropolar
                    Fuerza_de_retroceso_hetero_Magn_Z[j] += -8.0*(Wlm_ast(l,m,beta)*Wlm_ast(l-1,m,beta)*OmegaM_matriz_in_time_domain[l][m][j].real()*OmegaM_matriz_in_time_domain[l-1][m][j].real())*Pi_comp(l,m,m);
                    Fuerza_de_retroceso_hetero_Z[j] += Fuerza_de_retroceso_hetero_Elec_Z[j] + Fuerza_de_retroceso_hetero_Magn_Z[j];
                }
            }
        }
        Fuerza_de_retroceso_total_X[j]  = Fuerza_de_retroceso_homo_X[j] + Fuerza_de_retroceso_hetero_X[j];
        Fuerza_de_retroceso_total_Z[j]  = Fuerza_de_retroceso_homo_Z[j] + Fuerza_de_retroceso_hetero_Z[j];    
    }

    // Calcular los valores del dominio temporal
    double delta_t = 2*T / N;
    vector<double> time_domain_values(N);
    #pragma omp parallel for
    for (int s = 0; s < N; s++) {
        time_domain_values[s] = (-(N/2) + 1 + s)*delta_t;
    }

    //  Valores de las iteraciones correspondientes a los valores de t_min y t_max en la partición temporal
    int s_min = static_cast<int>(-1. + N * (1. + t_min/T)/2);
    int s_max = static_cast<int>(-1. + N * (1. + t_max/T)/2);

    //cout << "Error al generar el archivo .txt" << endl;

    // Construir el nombre del archivo dinámicamente
    string nombre_archivo = "Resultados_Fuerza_de_retroceso_n" + to_string(n) + "_T" + to_string(static_cast<int>(T)) + "_lmax" + to_string(l_max) + "_a" + to_string(static_cast<int>(a)) + "_b" + to_string(static_cast<int>(b)) + "_beta" + to_string(static_cast<int>(100.001*beta)) + ".txt";

    // Guardar los resultados en un archivo .txt
    ofstream file(nombre_archivo);
    for (int i = s_min; i < s_max; i++) {
        file << time_domain_values[i] << " " << Fuerza_de_retroceso_homo_X[i] << " " << Fuerza_de_retroceso_homo_Z[i] << " " 
                                             << Fuerza_de_retroceso_hetero_Elec_X[i] << " " << Fuerza_de_retroceso_hetero_Elec_Z[i] << " " 
                                             << Fuerza_de_retroceso_hetero_Magn_X[i] << " " << Fuerza_de_retroceso_hetero_Magn_Z[i] << " " 
                                             << Fuerza_de_retroceso_hetero_X[i] << " " << Fuerza_de_retroceso_hetero_Z[i] << " " 
                                             << Fuerza_de_retroceso_total_X[i] << " " << Fuerza_de_retroceso_total_Z[i] << "\n";
    }
    file.close();

    auto stop = chrono::high_resolution_clock::now();   //  Para el cronómetro de ejecución de cálculos
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start)/(60*pow(10.,6.));

    cout << "Lo resultados fueron guardados en " << nombre_archivo << endl;
    cout << "Tiempo de cómputo: " << duration.count() << " minutos" << endl;
    cout << "Se generaron " << contador << " transformadas de fourier" << endl;

    return 0;
}