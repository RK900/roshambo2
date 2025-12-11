//
// Created by dave on 04/12/2025.
//

#ifndef _ROSHAMBO2_CPP_HELPER_FUNCTIONS_H
#define _ROSHAMBO2_CPP_HELPER_FUNCTIONS_H

using DTYPE = float;

////////////////////////////////////////////////////////////////////////////////
/// Math helper functions
////////////////////////////////////////////////////////////////////////////////

/// @brief division of std::array by scalar
/// @tparam T
/// @tparam N
/// @param arr
/// @param scalar
/// @return
template<typename T, size_t N>
std::array<T, N> operator/(const std::array<T, N>& arr, const T& scalar);

/// @brief product of std::array by scalar
/// @tparam T
/// @tparam N
/// @param arr
/// @param scalar
/// @return
template<typename T, size_t N>
std::array<T, N> operator*(const T& scalar, const std::array<T, N>& arr);

/// @brief addition of std::arrays
/// @tparam T
/// @tparam N
/// @param arr1
/// @param arr2
/// @return
template<typename T, size_t N>
std::array<T, N> operator+(const std::array<T, N>& arr1, const std::array<T, N>& arr2);

/// @brief product of std::arrays
/// @tparam T
/// @tparam N
/// @param arr1
/// @param arr2
/// @return
template<typename T, size_t N>
std::array<T, N> operator*(const std::array<T, N>& arr1, const std::array<T, N>& arr2);

/// @brief 3x3 matrix x vector
/// @param mat matrix[3,3]
/// @param vec 3 vector[3]
/// @param result 3 vector[3]
void matvec3x3x3(DTYPE mat[][3], const DTYPE * vec, DTYPE * result);

/// @brief 3x3 matrix x vector
/// @param mat matrix[3,3]
/// @param vec 3 vector[3]
/// @return result 3 vector[3]
std::array<DTYPE,3> matvec3x3x3(const std::array<std::array<DTYPE,3>,3> &mat, const DTYPE * vec);

/// @brief 3x3 matrix x vector
/// @param mat matrix[3,3]
/// @param vec 3 vector[3]
/// @param result 3 vector[3]
void matvec3x3x3(const std::array<std::array<DTYPE,3>,3> &mat, const DTYPE * vec, DTYPE * result);

/// @brief convert quaternion to rotation matrix
/// @param q quaternion[4]
/// @param M matrix[3,3]
void quaternion_to_rotation_matrix(std::array<DTYPE,4> &q, DTYPE M[3][3]);

/// @brief transform molB by mat
/// @param mat matrix[3,3]
/// @param molB molecular coordinates[N,3]
/// @param molBT molecular coordinates[N,3]
/// @param NmolB N
void transform(DTYPE mat[][3], const DTYPE * molB, DTYPE * molBT, int NmolB);

/// @brief translate molA to molAT by t
/// @param t vector[3]
/// @param molA molecular coordinates[N,3]
/// @param molAT molecular coordinates[N,3]
/// @param NmolA N
void translate(DTYPE t[3], const DTYPE * molA, DTYPE * molAT, int NmolA);

///////////////////////////////////////////////////////////////////////////////
/// Volume functions
///////////////////////////////////////////////////////////////////////////////

/// @brief shape overlap volume of molA and molB
/// @param molA
/// @param NmolA
/// @param molB
/// @param NmolB
/// @return volume
DTYPE volume(const DTYPE * molA, int NmolA, const DTYPE * molB, int NmolB);

/// @brief color overlap volume of molA and molB
/// @param molA
/// @param NmolA
/// @param molA_type
/// @param molB
/// @param NmolB
/// @param molB_type
/// @param rmat
/// @param pmat
/// @param N_features
/// @return volume
DTYPE volume_color(const DTYPE * molA, int NmolA, const int * molA_type,
                   const DTYPE * molB, int NmolB, const int * molB_type,
                   const DTYPE * rmat, const DTYPE * pmat, int N_features);

////////////////////////////////////////////////////////////////////////////////
/// gradient functions
////////////////////////////////////////////////////////////////////////////////

/// @brief compute overlap gradients of molA and molB w.r.t molB
/// @param molA
/// @param NmolA
/// @param molB
/// @param NmolB
/// @return array[7] containing the quaternion gradients and the position gradients
std::array<DTYPE,7> get_gradient(const DTYPE * molA, int NmolA, const DTYPE * molB, int NmolB);

/// @brief compute overlap color gradients of molA and molB w.r.t molB
/// @param molA
/// @param NmolA
/// @param molA_type
/// @param molB
/// @param NmolB
/// @param molB_type
/// @param rmat
/// @param pmat
/// @param N_features
/// @return array[7] containing the quaternion gradients and the position gradients
std::array<DTYPE,7> get_gradient_color(const DTYPE * molA, int NmolA, const int * molA_type,
                                       const DTYPE * molB, int NmolB, const int * molB_type,
                                       const DTYPE * rmat, const DTYPE * pmat, int N_features);

////////////////////////////////////////////////////////////////////////////////
/// Optimization functions
////////////////////////////////////////////////////////////////////////////////


// Adagrad optimization step
void adagrad_step(std::array<DTYPE,4> &q, std::array<DTYPE,3> &t,  std::array<DTYPE,7> g,
                  std::array<DTYPE,7> &cache, DTYPE lr_q, DTYPE lr_t);

void single_conformer_optimiser(const DTYPE *molA, DTYPE *molAT, const int *molA_type,
                                int NmolA, int NmolA_real, int NmolA_color,
                                DTYPE self_overlap_A, DTYPE self_overlap_A_color,
                                const DTYPE *molB, DTYPE *molBT, const int *molB_type,
                                int NmolB, int NmolB_real, int NmolB_color,
                                const DTYPE *rmat, const DTYPE *pmat, int N_features,
                                bool optim_color, DTYPE mixing_param, DTYPE lr_q, DTYPE lr_t, int nsteps,
                                DTYPE *scores);

#endif //_ROSHAMBO2_CPP_HELPER_FUNCTIONS_H