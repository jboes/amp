!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      ! Fortran Version = 6
      subroutine check_version(version, warning) 
              implicit none
              integer :: version, warning
!f2py         intent(in) :: version
!f2py         intent(out) :: warning
              if (version .NE. 6) then
                warning = 1
              else
                warning = 0
              end if     
       end subroutine
       
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
       subroutine calculate_g2(numbers, rs, g_number, g_eta, cutoff, &
                                                       home, n, ridge)
               
              implicit none
              integer, dimension(n) :: numbers
              integer, dimension(1) :: g_number
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home
              integer :: n
              double precision ::  g_eta, cutoff
              double precision :: ridge
!f2py         intent(in) :: numbers, rs, g_number
!f2py         intent(in) :: g_eta, cutoff, home
!f2py         intent(hide) :: n
!f2py         intent(out) :: ridge
              integer :: j, match, xyz
              double precision, dimension(3) :: Rij_
              double precision :: Rij, term

              ridge = 0.0d0
              do j = 1, n
                  match = compare(numbers(j), g_number(1))
                  if (match == 1) then
                    do xyz = 1, 3
                      Rij_(xyz) = rs(j, xyz) - home(xyz)
                    end do
                    Rij = sqrt(dot_product(Rij_, Rij_))
                    term = exp(-g_eta*(Rij**2.0d0) / (cutoff ** 2.0d0))
                    term = term * cutoff_fxn(Rij, cutoff)
                    ridge = ridge + term
                  end if
                end do

      CONTAINS

      function compare(try, val) result(match)
!     Returns 1 if try is the same set as val, 0 if not.
              implicit none
              integer, intent(in) :: try, val
              integer :: match
              if (try == val) then
                      match = 1
              else
                      match = 0
              end if
      end function compare

      function cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, cutoff_fxn, pi
              if (r > cutoff) then
                      cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      cutoff_fxn = 0.5d0 * (cos(pi*r/cutoff) + 1.0d0)
              end if

      end function
      
      end subroutine calculate_g2
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       
      subroutine calculate_g4(numbers, rs, g_numbers, g_gamma, g_zeta, &
                           g_eta, cutoff, home, n, ridge)
               
              implicit none
              integer, dimension(n) :: numbers
              integer, dimension(2) :: g_numbers
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home
              integer :: n
              double precision :: g_gamma, g_zeta, g_eta, cutoff
              double precision :: ridge
!f2py         intent(in) :: numbers, rs, g_numbers, g_gamma, g_zeta
!f2py         intent(in) :: g_eta, cutoff, home
!f2py         intent(hide) :: n
!f2py         intent(out) :: ridge
              integer :: j, k, match, xyz
              double precision, dimension(3) :: Rij_, Rik_, Rjk_
              double precision :: Rij, Rik, Rjk, costheta, term

              ridge = 0.0d0
              do j = 1, n
                do k = (j + 1), n
                  match = compare(numbers(j), numbers(k), g_numbers(1),&
                              g_numbers(2))
                  if (match == 1) then
                    do xyz = 1, 3
                      Rij_(xyz) = rs(j, xyz) - home(xyz)
                      Rik_(xyz) = rs(k, xyz) - home(xyz)
                      Rjk_(xyz) = rs(k, xyz) - rs(j, xyz)
                    end do
                    Rij = sqrt(dot_product(Rij_, Rij_))
                    Rik = sqrt(dot_product(Rik_, Rik_))
                    Rjk = sqrt(dot_product(Rjk_, Rjk_))
                    costheta = dot_product(Rij_, Rik_) / Rij / Rik
                    term = (1.0d0 + g_gamma * costheta)**g_zeta
                    term = term*&
                    exp(-g_eta*(Rij**2 + Rik**2 + Rjk**2)&
                    /(cutoff ** 2.0d0))
                    term = term*(1.0d0/3.0d0)*(cutoff_fxn(Rij, cutoff)+&
                    cutoff_fxn(Rik, cutoff)+cutoff_fxn(Rjk, cutoff))
                    ridge = ridge + term
                  end if
                end do
              end do
              ridge = ridge * 2.0d0**(1.0d0 - g_zeta)


      CONTAINS

      function compare(try1, try2, val1, val2) result(match)
!     Returns 1 if (try1, try2) is the same set as (val1, val2), 0 if not.
              implicit none
              integer, intent(in) :: try1, try2, val1, val2
              integer :: match
              integer :: ntry1, ntry2, nval1, nval2
              ! First sort to avoid endless logical loops.
              if (try1 < try2) then
                      ntry1 = try1
                      ntry2 = try2
              else
                      ntry1 = try2
                      ntry2 = try1
              end if
              if (val1 < val2) then
                      nval1 = val1
                      nval2 = val2
              else
                      nval1 = val2
                      nval2 = val1
              end if
              if (ntry1 == nval1 .AND. ntry2 == nval2) then
                      match = 1
              else
                      match = 0
              end if

      end function compare

      function cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, cutoff_fxn, pi
              if (r > cutoff) then
                      cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      cutoff_fxn = 0.5d0 * (cos(pi*r/cutoff) + 1.0d0)
              end if

      end function
      
      end subroutine calculate_g4
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          
       subroutine calculate_der_g2(n_indices, numbers, rs, g_number,&
                             g_eta, cutoff, aa, home, mm, ii, n, ridge)
               
              implicit none
              integer, dimension(n) :: n_indices, numbers
              integer, dimension(1) :: g_number
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home, Rj
              integer :: n, mm, ii, aa
              double precision ::  g_eta, cutoff
              double precision :: ridge
!f2py         intent(in) :: n_indices, numbers, rs, g_number
!f2py         intent(in) :: g_eta, cutoff, aa, home, mm, ii
!f2py         intent(hide) :: n
!f2py         intent(out) :: ridge
              integer :: j, match, xyz
              double precision, dimension(3) :: Raj_
              double precision :: Raj, term, term1, term2

              ridge = 0.0d0
              do j = 1, n
                  match = compare(numbers(j), g_number(1))
                  if (match == 1) then
                    do xyz = 1, 3
                      Rj(xyz) = rs(j, xyz)
                      Raj_(xyz) = home(xyz) - Rj(xyz)
                    end do
                    Raj = sqrt(dot_product(Raj_, Raj_))
                    term1 = - 2.0d0 * g_eta * Raj * & 
                    cutoff_fxn(Raj, cutoff) / (cutoff ** 2.0d0) + &
                    der_cutoff_fxn(Raj, cutoff)
                    term2 = &
                     der_position(aa, n_indices(j), home, Rj, mm, ii)
                    term = exp(- g_eta * (Raj**2.0d0) / &
                    (cutoff ** 2.0d0)) * term1 * term2
                    ridge = ridge + term
                  end if
                end do

      CONTAINS

      function compare(try, val) result(match)
!     Returns 1 if try is the same set as val, 0 if not.
              implicit none
              integer, intent(in) :: try, val
              integer :: match
              if (try == val) then
                      match = 1
              else
                      match = 0
              end if
      end function compare
      
      function cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, cutoff_fxn, pi
              if (r > cutoff) then
                      cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      cutoff_fxn = 0.5d0 * (cos(pi*r/cutoff) + 1.0d0)
              end if
      end function
      
      function der_cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, der_cutoff_fxn, pi
              if (r > cutoff) then
                      der_cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      der_cutoff_fxn = -0.5d0 * pi * sin(pi*r/cutoff) &
                       / cutoff
              end if
      end function
      
      function der_position(mm, nn, Rm, Rn, ll, ii)
              integer mm, nn, ll, ii
              double precision, dimension(3) :: Rm, Rn, Rmn_
              double precision :: der_position, Rmn
              do xyz = 1, 3
                      Rmn_(xyz) = Rm(xyz) - Rn(xyz)
              end do
              Rmn = sqrt(dot_product(Rmn_, Rmn_))
              if ((ll == mm) .AND. (mm /= nn)) then
                      der_position = (Rm(ii + 1) - Rn(ii + 1)) / Rmn
              else if ((ll == nn) .AND. (mm /= nn)) then
                      der_position = - (Rm(ii + 1) - Rn(ii + 1)) / Rmn
              else
                      der_position = 0.0d0
              end if
      end function
      
      end subroutine calculate_der_g2
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine calculate_der_g4(n_indices, numbers, rs, g_numbers, &
                            g_gamma, g_zeta, &
                            g_eta, cutoff, aa, home, mm, ii, n, ridge)
               
              implicit none
              integer, dimension(n) :: n_indices, numbers
              integer, dimension(2) :: g_numbers
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home, Rj, Rk
              integer :: n, aa, mm, ii
              double precision :: g_gamma, g_zeta, g_eta, cutoff
              double precision :: ridge
!f2py         intent(in) :: numbers, rs, g_numbers, g_gamma, g_zeta
!f2py         intent(in) :: g_eta, cutoff, home, n_indices , aa, mm, ii
!f2py         intent(hide) :: n
!f2py         intent(out) :: ridge
              integer :: j, k, match, xyz
              double precision, dimension(3) :: Raj_, Rak_, Rjk_
             double precision :: Raj, Rak, Rjk, costheta, c1, c2, c3, c4
              double precision :: term1, term2, term3, term4, term5
              double precision :: term6, term7, term8, term9, term10
              double precision :: term11, term

              ridge = 0.0d0
              do j = 1, n
                do k = (j + 1), n
                  match = compare(numbers(j), numbers(k), g_numbers(1),&
                                 g_numbers(2))
                  if (match == 1) then
                    do xyz = 1, 3
                      Rj(xyz) = rs(j, xyz)
                      Rk(xyz) = rs(k, xyz)
                      Raj_(xyz) = Rj(xyz) - home(xyz)
                      Rak_(xyz) = Rk(xyz) - home(xyz)
                      Rjk_(xyz) = Rj(xyz) - Rk(xyz)
                    end do
                    Raj = sqrt(dot_product(Raj_, Raj_))
                    Rak = sqrt(dot_product(Rak_, Rak_))
                    Rjk = sqrt(dot_product(Rjk_, Rjk_))
                    costheta = dot_product(Raj_, Rak_) / Raj / Rak
                    c1 = (1.0d0 + g_gamma * costheta)
                    c2 = cutoff_fxn(Raj, cutoff)
                    c3 = cutoff_fxn(Rak, cutoff)
                    c4 = cutoff_fxn(Rjk, cutoff)
                    if (g_zeta == 1) then
                        term1 = exp(-g_eta*(Raj**2 + Rak**2 + Rjk**2)&
                        / (cutoff ** 2.0d0))
                    else
                        term1 = (c1**(g_zeta - 1.0d0)) &
                             * exp(-g_eta*(Raj**2 + Rak**2 + Rjk**2)&
                             / (cutoff ** 2.0d0))
                    end if
                    term2 = (1.0d0/3.0d0) * (c2 + c3 + c4)
                    term3 = der_cos_theta(aa, n_indices(j), &
                     n_indices(k), home, Rj, Rk, mm, ii)
                    term4 = g_gamma * g_zeta * term3
                    term5 = &
                    der_position(aa, n_indices(j), home, Rj, mm, ii)
                    term4 = term4 - 2.0d0 * c1 * g_eta * Raj * term5&
                                    / (cutoff ** 2.0d0)
                    term6 = &
                    der_position(aa, n_indices(k), home, Rk, mm, ii)
                    term4 = term4 - 2.0d0 * c1 * g_eta * Rak * term6&
                                    / (cutoff ** 2.0d0)
                    term7 =  der_position(n_indices(j), n_indices(k),&
                                                      Rj, Rk, mm, ii)
                    term4 = term4 - 2.0d0 * c1 * g_eta * Rjk * term7&
                                    / (cutoff ** 2.0d0)
                    term2 = term2 * term4
                    term8 = c1 * (1.0d0/3.0d0) * &
                    der_cutoff_fxn(Raj, cutoff) * term5
                    term9 = c1 * (1.0d0/3.0d0) * &
                    der_cutoff_fxn(Rak, cutoff) * term6
                    term10 = c1 * (1.0d0/3.0d0) * &
                    der_cutoff_fxn(Rjk, cutoff) * term7
                    term11 = term2 + term8 + term9 + term10
                    term = term1 * term11
                    ridge = ridge + term
                  end if
                end do
              end do
              ridge = ridge * (2.0d0**(1.0d0 - g_zeta))

      CONTAINS

      function compare(try1, try2, val1, val2) result(match)
!     Returns 1 if (try1, try2) is the same set as (val1, val2), 0 if not.
              implicit none
              integer, intent(in) :: try1, try2, val1, val2
              integer :: match
              integer :: ntry1, ntry2, nval1, nval2
              ! First sort to avoid endless logical loops.
              if (try1 < try2) then
                      ntry1 = try1
                      ntry2 = try2
              else
                      ntry1 = try2
                      ntry2 = try1
              end if
              if (val1 < val2) then
                      nval1 = val1
                      nval2 = val2
              else
                      nval1 = val2
                      nval2 = val1
              end if
              if (ntry1 == nval1 .AND. ntry2 == nval2) then
                      match = 1
              else
                      match = 0
              end if
      end function compare

      function cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, cutoff_fxn, pi
              if (r > cutoff) then
                      cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      cutoff_fxn = 0.5d0 * (cos(pi*r/cutoff) + 1.0d0)
              end if
      end function
      
      function der_cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, der_cutoff_fxn, pi
              if (r > cutoff) then
                      der_cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      der_cutoff_fxn = -0.5d0 * pi * sin(pi*r/cutoff) &
                           / cutoff
              end if
      end function
      
      function der_position(mm, nn, Rm, Rn, ll, ii)
              integer :: mm, nn, ll, ii
              double precision, dimension(3) :: Rm, Rn, Rmn_
              double precision :: der_position, Rmn
              do xyz = 1, 3
                      Rmn_(xyz) = Rm(xyz) - Rn(xyz)
              end do
              Rmn = sqrt(dot_product(Rmn_, Rmn_))
              if ((ll == mm) .AND. (mm /= nn)) then
                      der_position = (Rm(ii + 1) - Rn(ii + 1)) / Rmn
              else if ((ll == nn) .AND. (mm /= nn)) then
                      der_position = - (Rm(ii + 1) - Rn(ii + 1)) / Rmn
              else
                      der_position = 0.0d0
              end if
      end function
      
      function der_cos_theta(aa, jj, kk, home, Rj, Rk, mm, ii)
      implicit none
      integer :: aa, jj, kk, mm, ii
      double precision :: der_cos_theta
      double precision, dimension(3) :: home, Rj, Rk
      
      do xyz = 1, 3
            Raj_(xyz) = home(xyz) - Rj(xyz)
            Rak_(xyz) = home(xyz) - Rk(xyz)
      end do
      Raj = sqrt(dot_product(Raj_, Raj_))
      Rak = sqrt(dot_product(Rak_, Rak_))
      der_cos_theta = 1.0d0 / (Raj * Rak) * &
             dot_product(der_position_vector(aa, jj, mm, ii), Rak_)
      der_cos_theta =  der_cos_theta + 1.0d0 / (Raj * Rak) * &
             dot_product(der_position_vector(aa, kk, mm, ii), Raj_)
      der_cos_theta =  der_cos_theta - 1.0d0 / (Raj * Raj * Rak) * &
             dot_product(Raj_, Rak_) * &
             der_position(aa, jj, home, Rj, mm, ii)
      der_cos_theta =  der_cos_theta - 1.0d0 / (Raj * Rak * Rak) * &
             dot_product(Raj_, Rak_) * &
             der_position(aa, kk, home, Rk, mm, ii)
     
      end function
       
      function der_position_vector(aa, bb, mm, ii)
      implicit none
      integer :: aa, bb, mm, ii
      integer, dimension(3) :: der_position_vector
      
      der_position_vector(1) = (Kronecker_delta(mm, aa) &
          - Kronecker_delta(mm, bb)) * Kronecker_delta(0, ii)
      der_position_vector(2) = (Kronecker_delta(mm, aa) &
          - Kronecker_delta(mm, bb)) * Kronecker_delta(1, ii)
      der_position_vector(3) = (Kronecker_delta(mm, aa) &
          - Kronecker_delta(mm, bb)) * Kronecker_delta(2, ii)
    
      end function
       
      function Kronecker_delta(ii, jj)
      implicit none
      integer :: ii, jj
      integer :: Kronecker_delta
      
      if (ii == jj) then
        Kronecker_delta = 1
      else
        Kronecker_delta = 0
      end if
    
      end function

      end subroutine calculate_der_g4
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      subroutine calculate_error(no_atom, nn_values, real_values,&
                 no_values, rmse_per_atom, square_error)

      implicit none
      double precision:: rmse_per_atom, square_error
      integer :: no_atom, no_values
      double precision, dimension (no_values) :: nn_values, real_values
             
!f2py   intent (in) :: no_atom, no_values, nn_values, real_values
!f2py  intent (out) :: rmse_per_atom, square_error

      integer :: i
      double precision :: square_error_per_image
      square_error = 0.0d0
      do i = 1, no_values
      square_error = square_error + &
      (nn_values(i) - real_values(i))**(2.0d0)
      enddo
      square_error_per_image = square_error / no_values
      rmse_per_atom = sqrt(square_error_per_image) / no_atom
      end subroutine calculate_error
      
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      module image_energy_data

      integer :: no_of_elements, no_of_atoms
      integer, allocatable :: elements_numbers(:)
      integer, allocatable :: no_layers_of_elements(:)
      integer, allocatable :: no_nodes_of_elements(:)
      integer, allocatable :: atomic_numbers_of_image(:)
      double precision, allocatable :: raveled_fps_of_image(:, :)
      integer, allocatable :: len_fingerprints_of_elements(:)
      double precision, allocatable ::min_fingerprints(:, :)
      double precision, allocatable ::max_fingerprints(:, :)

      end module image_energy_data

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_potential_energy(&
        len_of_variables, variables, &
        activation, nn_energy)
                        
      use image_energy_data
      implicit none

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! input/output variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      character(len=7) :: activation
      integer :: len_of_variables
      double precision :: variables(len_of_variables)
      double precision :: nn_energy
!f2py         intent(in) :: variables , len_of_variables, activation
!f2py         intent(out) :: nn_energy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! type definition !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type :: real_one_d_array_type
        sequence
        double precision, allocatable:: onedarray(:)
      end type real_one_d_array_type

      type :: real_two_d_array_type
        sequence
        double precision, allocatable:: twodarray(:,:)
      end type real_two_d_array_type
      
      type :: embedded_real_one_one_d_array_type
        sequence
        type(real_one_d_array_type), allocatable:: onedarray(:)
      end type embedded_real_one_one_d_array_type

      type :: element_variables_type
        sequence
        double precision :: scaling_intercept
        double precision :: scaling_slope
        type(real_two_d_array_type), allocatable:: weights(:)
      end type element_variables_type

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! dummy variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type(real_one_d_array_type) :: &
      unraveled_fingerprints(no_of_atoms)
      type(element_variables_type) :: &
      unraveled_variables(no_of_elements)
      integer :: index, j, m, n, p, layer, k, l, no_of_rows, &
      no_of_cols, symbol, element, atom

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! calculations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call unravel_variables()
      call unravel_fingerprints()
      call scale_fingerprints()

      nn_energy = calculate_nn_energy(&
      unraveled_fingerprints, unraveled_variables)

      !deallocations
      do element = 1, size(unraveled_variables)
        do layer = 1, size(unraveled_variables(element)%weights)
            deallocate(&
            unraveled_variables(element)%weights(layer)%twodarray)
        end do
        deallocate(unraveled_variables(element)%weights)
      end do

      contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine unravel_variables()
 
      k = 0
      l = 0
      do symbol = 1, no_of_elements
        allocate(unraveled_variables(symbol)%weights(&
        no_layers_of_elements(symbol)-1))
        if (symbol .GT. 1) then
            k = k + no_layers_of_elements(symbol - 1)
        end if
        do j = 1, no_layers_of_elements(symbol) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_variables(symbol)%weights(j)%twodarray(&
            no_of_rows, no_of_cols))
            do m = 1, no_of_rows
                do n = 1, no_of_cols
                    unraveled_variables(symbol)%weights(j)%twodarray(&
                    m, n) = variables(l + (m - 1) * no_of_cols + n)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do symbol = 1, no_of_elements
        unraveled_variables(symbol)%scaling_intercept = &
        variables(l + 2 *  symbol - 1)
        unraveled_variables(symbol)%scaling_slope = &
        variables(l + 2 * symbol)
      end do
      end subroutine unravel_variables
 
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_fingerprints()    

      do index = 1, no_of_atoms
          do symbol = 1, no_of_elements
              if (atomic_numbers_of_image(index) &
              == elements_numbers(symbol)) then
                  allocate(unraveled_fingerprints(index)&
                  %onedarray(len_fingerprints_of_elements(symbol))) 
                  exit
              end if
          end do    
          do l = 1, len_fingerprints_of_elements(symbol)
              unraveled_fingerprints(index)&
              %onedarray(l) = raveled_fps_of_image(index, l)
          end do
      end do
      
      end subroutine unravel_fingerprints


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      function calculate_nn_energy(&
      unraveled_fingerprints, unraveled_variables)
 
      type(real_one_d_array_type) :: unraveled_fingerprints(no_of_atoms)
      type(element_variables_type) :: &
      unraveled_variables(no_of_elements)
      double precision :: calculate_nn_energy
      type(embedded_real_one_one_d_array_type) :: &
      o(no_of_atoms), ohat(no_of_atoms)
      double precision :: atomic_nn_energy
      integer, allocatable :: hiddensizes(:)
      type(real_one_d_array_type), allocatable :: atomic_net(:)
 
      calculate_nn_energy = 0.0d0
      do index = 1, no_of_atoms
        p = 0
        do symbol = 1, no_of_elements
            if (atomic_numbers_of_image(index) ==&
            elements_numbers(symbol)) then
                exit
            else 
                p = p + no_layers_of_elements(symbol)
            end if
        end do
        allocate(hiddensizes(no_layers_of_elements(symbol) - 2))
        do m = 1, no_layers_of_elements(symbol) - 2
            hiddensizes(m) = no_nodes_of_elements(p + m + 1)
        end do
        allocate(o(index)%onedarray(no_layers_of_elements(symbol)))
        allocate(ohat(index)%onedarray(no_layers_of_elements(symbol)))
        allocate(atomic_net(no_layers_of_elements(symbol)))
        layer = 1
        allocate(o(index)%onedarray(1)%onedarray(size(&
        unraveled_fingerprints(index)%onedarray)))
        allocate(ohat(index)%onedarray(1)%onedarray(&
        size(unraveled_fingerprints(index)%onedarray) + 1))
        do m = 1, size(unraveled_variables(symbol)&
        %weights(1)%twodarray, dim=1) - 1
            o(index)%onedarray(1)%onedarray(m) = &
            unraveled_fingerprints(index)%onedarray(m)
        end do
        do layer = 1, size(hiddensizes) + 1
            do m = 1, size(unraveled_variables(symbol)&
            %weights(layer)%twodarray, dim=1) - 1
                ohat(index)%onedarray(layer)%onedarray(m) = &
                o(index)%onedarray(layer)%onedarray(m)
            end do
            ohat(index)%onedarray(layer)%onedarray(&
            size(unraveled_variables(symbol)&
            %weights(layer)%twodarray, dim=1)) = 1.0d0
            allocate(atomic_net(layer)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)))
            allocate(o(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)))
            allocate(ohat(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2) + 1))
            do m = 1, size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)
                atomic_net(layer)%onedarray(m) = 0.0d0
                do n = 1, size(unraveled_variables(symbol)%&
                weights(layer)%twodarray, dim=1)
                    atomic_net(layer)%onedarray(m) =  &
                    atomic_net(layer)%onedarray(m) + &
                    ohat(index)%onedarray(layer)%onedarray(n) * &
                    unraveled_variables(symbol)&
                    %weights(layer)%twodarray(n, m)
                end do
                if (activation == 'tanh') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    tanh(atomic_net(layer)%onedarray(m))
                else if (activation == 'sigmoid') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    1. / (1. +  exp(- atomic_net(layer)%onedarray(m)))
                else if (activation == 'linear') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    atomic_net(layer)%onedarray(m)
                else
                    print *, &
                    "Fortran Error: Unknown activation function!"
                end if
                ohat(index)%onedarray(layer + 1)%onedarray(m) = &
                o(index)%onedarray(layer + 1)%onedarray(m)
            end do
            ohat(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%weights(&
            layer)%twodarray, dim=2) + 1) =  1.0
        end do
        atomic_nn_energy = unraveled_variables(symbol)&
        %scaling_slope * o(index)%onedarray(&
        layer)%onedarray(1) + unraveled_variables(&
        symbol)%scaling_intercept
        calculate_nn_energy = calculate_nn_energy +  atomic_nn_energy
        deallocate(hiddensizes)
        deallocate(atomic_net)
        end do

      !deallocations
      do atom = 1, no_of_atoms
          deallocate(unraveled_fingerprints(atom)%onedarray)
      end do
      do atom = 1, no_of_atoms
          do layer = 1, size(o(atom)%onedarray)
              deallocate(o(atom)%onedarray(layer)%onedarray)
          end do
          deallocate(o(atom)%onedarray)
      end do
      do atom = 1, no_of_atoms
          do layer = 1, size(ohat(atom)%onedarray)
              deallocate(ohat(atom)%onedarray(layer)%onedarray)
          end do
          deallocate(ohat(atom)%onedarray)
      end do

      end function calculate_nn_energy
 
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine scale_fingerprints() 
      
      double precision :: double_temp
  
      do index = 1, size(unraveled_fingerprints)
          do symbol = 1, no_of_elements
              if (atomic_numbers_of_image(index)&             
               == elements_numbers(symbol)) then
                  exit
              end if
          end do    
          do l = 1, len_fingerprints_of_elements(symbol)
              if ((max_fingerprints(symbol, l) - &
              min_fingerprints(symbol, l)) .GT. (10.0d0 ** (-8.0d0)))&
               then
                  double_temp = &
                  unraveled_fingerprints(index)%onedarray(l)
                  double_temp = -1.0d0 + 2.0d0 * (double_temp - &
                  min_fingerprints(symbol, l)) / &
                  (max_fingerprints(symbol, l) - &
                  min_fingerprints(symbol, l))
                  unraveled_fingerprints(index)%onedarray(l) = &
                  double_temp
              endif
          end do
      end do

      end subroutine scale_fingerprints
 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      end subroutine get_potential_energy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      module image_force_data

      integer :: no_of_elements, no_of_atoms
      integer, allocatable :: elements_numbers(:)
      integer, allocatable :: no_layers_of_elements(:)
      integer, allocatable :: no_nodes_of_elements(:)
      integer, allocatable :: atomic_numbers_of_image(:)
      double precision, allocatable :: raveled_fps_of_image(:, :)
      integer, allocatable :: len_fingerprints_of_elements(:)
      integer, allocatable :: list_of_no_of_neighbors(:)
      integer, allocatable :: raveled_neighborlists(:)
      double precision, allocatable :: raveled_der_fingerprints(:, :)
      double precision, allocatable ::min_fingerprints(:, :)
      double precision, allocatable ::max_fingerprints(:, :)

      end module image_force_data

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_forces(no_of_atoms_, activation, &
        len_of_variables, variables, nn_forces)
                        
      use image_force_data                        
      implicit none

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! input/output variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      character(len=7) :: activation
      integer :: len_of_variables, no_of_atoms_
      double precision :: variables(len_of_variables)
      double precision :: nn_forces(no_of_atoms_, 3)
!f2py         intent(in) :: no_of_atoms_, variables , len_of_variables, activation
!f2py         intent(out) :: nn_forces

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! type definition !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type :: real_one_d_array_type
        sequence
        double precision, allocatable:: onedarray(:)
      end type real_one_d_array_type

      type :: integer_one_d_array_type
        sequence
        integer, allocatable:: onedarray(:)
      end type integer_one_d_array_type

      type :: real_two_d_array_type
        sequence
        double precision, allocatable:: twodarray(:,:)
      end type real_two_d_array_type
      
       type :: embedded_real_one_two_d_array_type
        sequence
        type(real_two_d_array_type), allocatable:: onedarray(:)
      end type embedded_real_one_two_d_array_type

      type :: embedded_real_one_one_d_array_type
        sequence
        type(real_one_d_array_type), allocatable:: onedarray(:)
      end type embedded_real_one_one_d_array_type

      type :: element_variables_type
        sequence
        double precision :: scaling_intercept
        double precision :: scaling_slope
        type(real_two_d_array_type), allocatable:: weights(:)
      end type element_variables_type

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! dummy variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type(real_one_d_array_type) :: &
      unraveled_fingerprints(no_of_atoms)
      type(element_variables_type) :: &
      unraveled_variables(no_of_elements)
      integer :: i, index, j, m, n, p, layer, nn, k, q, l, no_of_rows, &
      no_of_cols, symbol, element, atom
      type(integer_one_d_array_type) :: &
      unraveled_neighborlists(no_of_atoms)
      type(embedded_real_one_two_d_array_type) :: &
      unraveled_der_fingerprints(no_of_atoms)
      integer :: n_index, self_index

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! calculations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call unravel_variables()
      call unravel_fingerprints()
      call scale_fingerprints()
      call unravel_neighborlists()
      call unravel_der_fingerprints()
      call scale_der_fingerprints()
      call calculate_nn_forces(unraveled_variables)
        
      !deallocations
      do atom = 1, no_of_atoms
          deallocate(unraveled_fingerprints(atom)%onedarray)
      end do
      do element = 1, size(unraveled_variables)
        do layer = 1, size(unraveled_variables(element)%weights)
            deallocate(&
            unraveled_variables(element)%weights(layer)%twodarray)
        end do
        deallocate(unraveled_variables(element)%weights)
      end do
      do self_index = 1, no_of_atoms
          do n_index = 1, &
              size(unraveled_der_fingerprints(self_index)%onedarray)
              deallocate(unraveled_der_fingerprints(&
              self_index)%onedarray(n_index)%twodarray)
          end do
          deallocate(unraveled_der_fingerprints(&
          self_index)%onedarray)
      end do
      do index = 1, no_of_atoms
          deallocate(unraveled_neighborlists(index)%onedarray)
      end do

      contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine unravel_variables()
 
      k = 0
      l = 0
      do symbol = 1, no_of_elements
        allocate(unraveled_variables(symbol)%weights(&
        no_layers_of_elements(symbol)-1))
        if (symbol .GT. 1) then
            k = k + no_layers_of_elements(symbol - 1)
        end if
        do j = 1, no_layers_of_elements(symbol) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_variables(symbol)%weights(j)%twodarray(&
            no_of_rows, no_of_cols))
            do m = 1, no_of_rows
                do n = 1, no_of_cols
                    unraveled_variables(symbol)%weights(j)%twodarray(&
                    m, n) = variables(l + (m - 1) * no_of_cols + n)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do symbol = 1, no_of_elements
        unraveled_variables(symbol)%scaling_intercept = &
        variables(l + 2 *  symbol - 1)
        unraveled_variables(symbol)%scaling_slope = &
        variables(l + 2 * symbol)
      end do
      end subroutine unravel_variables

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_der_fingerprints()

      integer :: no_of_neighbors, n_symbol, temp
      integer, allocatable :: n_self_indices(:)

      k = 0
      temp = 0
      do self_index = 1, no_of_atoms
          allocate(n_self_indices(size(unraveled_neighborlists(&
          self_index)%onedarray)))
          do p = 1, size(unraveled_neighborlists(self_index)%onedarray)
              n_self_indices(p) = &
              unraveled_neighborlists(self_index)%onedarray(p)
          end do
          no_of_neighbors = list_of_no_of_neighbors(self_index)
          allocate(unraveled_der_fingerprints(&
          self_index)%onedarray(no_of_neighbors))
          do n_index = 1, no_of_neighbors
              do n_symbol = 1, no_of_elements
              if (atomic_numbers_of_image(n_self_indices(n_index)) == &
              elements_numbers(n_symbol)) then
                  exit
              end if
              end do
              allocate(unraveled_der_fingerprints(&
              self_index)%onedarray(n_index)%twodarray&
              (3, len_fingerprints_of_elements(n_symbol)))
              do p = 1, 3
                  do q = 1, len_fingerprints_of_elements(n_symbol)
                      unraveled_der_fingerprints(&
                      self_index)%onedarray(n_index)%twodarray(p, q) &
                      = raveled_der_fingerprints(&
                      3 * temp + 3 * n_index + p - 3, q)
                  end do
              end do
          end do
          deallocate(n_self_indices)
          temp = temp + no_of_neighbors
      end do
      end subroutine unravel_der_fingerprints

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine unravel_neighborlists()

      integer :: temp

      temp = 0
      do index = 1, no_of_atoms
          allocate(unraveled_neighborlists(&
          index)%onedarray(list_of_no_of_neighbors(index)))
          do p = 1, list_of_no_of_neighbors(index)
              unraveled_neighborlists(index)%onedarray(p) = &
              raveled_neighborlists(temp + p) + 1
          end do
          temp = temp + list_of_no_of_neighbors(index)
      end do
      end subroutine unravel_neighborlists
 
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_fingerprints()    

      do index = 1, no_of_atoms
          do symbol = 1, no_of_elements
              if (atomic_numbers_of_image(index) &
              == elements_numbers(symbol)) then
                  allocate(unraveled_fingerprints(index)&
                  %onedarray(len_fingerprints_of_elements(symbol))) 
                  exit
              end if
          end do    
          do l = 1, len_fingerprints_of_elements(symbol)
              unraveled_fingerprints(index)&
              %onedarray(l) = raveled_fps_of_image(index, l)
          end do
      end do
      end subroutine unravel_fingerprints
 
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine calculate_nn_forces(unraveled_variables)
 
      type(element_variables_type) :: &
      unraveled_variables(no_of_elements)
      double precision :: temp1, temp2
     
      type(embedded_real_one_one_d_array_type) :: &
      o(no_of_atoms), ohat(no_of_atoms)
      type(embedded_real_one_one_d_array_type) :: delta(no_of_atoms)
      type(embedded_real_one_one_d_array_type) :: D(no_of_atoms)
      type (embedded_real_one_one_d_array_type), allocatable :: &
      der_coordinates_o(:, :)
      integer, allocatable :: n_self_indices(:)
      integer :: n_symbol, n_index
      double precision, allocatable :: temp(:)
      type(real_one_d_array_type), allocatable :: der_coordinates_D(:)
      type (real_one_d_array_type), allocatable :: &
      der_coordinates_delta(:)
      integer, allocatable :: hiddensizes(:)
      type(real_one_d_array_type), allocatable :: atomic_net(:)
      double precision, allocatable :: temp3(:), temp4(:), temp5(:), &
      temp6(:)

      do index = 1, no_of_atoms
        p = 0
        do symbol = 1, no_of_elements
            if (atomic_numbers_of_image(index) ==&
            elements_numbers(symbol)) then
                exit
            else 
                p = p + no_layers_of_elements(symbol)
            end if
        end do
        allocate(hiddensizes(no_layers_of_elements(symbol) - 2))
        do m = 1, no_layers_of_elements(symbol) - 2
            hiddensizes(m) = no_nodes_of_elements(p + m + 1)
        end do
        allocate(o(index)%onedarray(no_layers_of_elements(symbol)))
        allocate(ohat(index)%onedarray(no_layers_of_elements(symbol)))
        allocate(atomic_net(no_layers_of_elements(symbol)))
        layer = 1
        allocate(o(index)%onedarray(1)%onedarray(size(&
        unraveled_fingerprints(index)%onedarray)))
        allocate(ohat(index)%onedarray(1)%onedarray(&
        size(unraveled_fingerprints(index)%onedarray) + 1))
        do m = 1, size(unraveled_variables(symbol)&
        %weights(1)%twodarray, dim=1) - 1
            o(index)%onedarray(1)%onedarray(m) = &
            unraveled_fingerprints(index)%onedarray(m)
        end do
        do layer = 1, size(hiddensizes) + 1
            do m = 1, size(unraveled_variables(symbol)&
            %weights(layer)%twodarray, dim=1) - 1
                ohat(index)%onedarray(layer)%onedarray(m) = &
                o(index)%onedarray(layer)%onedarray(m)
            end do
            ohat(index)%onedarray(layer)%onedarray(&
            size(unraveled_variables(symbol)&
            %weights(layer)%twodarray, dim=1)) = 1.0d0
            allocate(atomic_net(layer)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)))
            allocate(o(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)))
            allocate(ohat(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2) + 1))
            do m = 1, size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)
                atomic_net(layer)%onedarray(m) = 0.0d0
                do n = 1, size(unraveled_variables(symbol)%&
                weights(layer)%twodarray, dim=1)
                    atomic_net(layer)%onedarray(m) =  &
                    atomic_net(layer)%onedarray(m) + &
                    ohat(index)%onedarray(layer)%onedarray(n) * &
                    unraveled_variables(symbol)&
                    %weights(layer)%twodarray(n, m)
                end do
                if (activation == 'tanh') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    tanh(atomic_net(layer)%onedarray(m))
                else if (activation == 'sigmoid') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    1. / (1. +  exp(- atomic_net(layer)%onedarray(m)))
                else if (activation == 'linear') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    atomic_net(layer)%onedarray(m)
                else
                    print *, &
                    "Fortran Error: Unknown activation function!"
                end if
                ohat(index)%onedarray(layer + 1)%onedarray(m) = &
                o(index)%onedarray(layer + 1)%onedarray(m)
            end do
            ohat(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%weights(&
            layer)%twodarray, dim=2) + 1) =  1.0
        end do
        deallocate(hiddensizes)
        deallocate(atomic_net)
        
        nn = size(o(index)%onedarray) - 2
        allocate(D(index)%onedarray(nn + 1))
        do layer = 1, nn + 1
            allocate(D(index)%onedarray(layer)%onedarray(&
            size(o(index)%onedarray(layer + 1)%onedarray)))
            do m = 1, size(o(index)%onedarray(layer + 1)%onedarray)
                if (activation == "tanh") then
                    D(index)%onedarray(layer)%onedarray(m) = &
                    (1.0d0 - o(index)%onedarray(layer + 1)%onedarray(m)&
                     * o(index)%onedarray(layer + 1)%onedarray(m))
                elseif (activation == "sigmoid") then
                    D(index)%onedarray(layer)%onedarray(m) = &
                    o(index)%onedarray(layer + 1)%onedarray(m) * &
                    (1.0d0 - o(index)%onedarray(layer + 1)%onedarray(m))
                elseif (activation == "linear") then
                    D(index)%onedarray(layer)%onedarray(m) = 1.0d0
                end if
            end do    
        end do
        allocate(delta(index)%onedarray(nn + 1))
        allocate(delta(index)%onedarray(nn + 1)%onedarray(1))
        delta(index)%onedarray(nn + 1)%onedarray(1) = &
        D(index)%onedarray(nn + 1)%onedarray(1)
        do layer = nn, 1, -1
            allocate(delta(index)%onedarray(layer)%onedarray(&
            size(D(index)%onedarray(layer)%onedarray)))
            do m = 1, size(D(index)%onedarray(layer)%onedarray)
                delta(index)%onedarray(layer)%onedarray(m) = 0.0d0
                do q = 1, &
                size(delta(index)%onedarray(layer + 1)%onedarray)
                    temp1 = D(index)%onedarray(layer)%onedarray(m) * &
                    unraveled_variables(symbol)%weights(layer + 1)&
                    %twodarray(m, q)
                    temp2 = temp1  * &
                    delta(index)%onedarray(layer + 1)%onedarray(q)
                    delta(index)%onedarray(layer)%onedarray(m) = &
                    delta(index)%onedarray(layer)%onedarray(m) + temp2
                end do
            end do
        end do
      end do
 
      do self_index = 1, no_of_atoms
          do i = 1, 3
              nn_forces(self_index, i) = 0.0d0
          end do
      end do
      do self_index = 1, no_of_atoms
          allocate(n_self_indices(size(unraveled_neighborlists(&
          self_index)%onedarray)))
          do p = 1, size(unraveled_neighborlists(self_index)%onedarray)
              n_self_indices(p) = unraveled_neighborlists(&
              self_index)%onedarray(p)
          end do 
          allocate(der_coordinates_o(size(n_self_indices), 3))
          do l = 1, size(n_self_indices)
              n_index = n_self_indices(l)
              do n_symbol = 1, no_of_elements
                  if (atomic_numbers_of_image(n_index) &
                  == elements_numbers(n_symbol)) then
                      exit
                  end if
              end do
              nn = size(o(n_index)%onedarray) - 2
              do i = 1, 3
                  allocate(der_coordinates_o(l, i)%onedarray(nn + 2))
                  allocate(der_coordinates_o(l, i)%onedarray(1)%&
                  onedarray(len_fingerprints_of_elements(n_symbol)))
                  do m = 1, size(unraveled_der_fingerprints&
                  (self_index)%onedarray(l)%twodarray, dim = 2)
                  der_coordinates_o(l, i)%onedarray(1)%onedarray(m)=&
                  unraveled_der_fingerprints(&
                  self_index)%onedarray(l)%twodarray(i, m)
                  end do
                  do layer = 1, nn + 1
                      allocate(temp(size(unraveled_variables(&
                      n_symbol)%weights(layer)%twodarray, dim = 2)))
                      do p = 1, &
                      size(unraveled_variables(n_symbol)&
                      %weights(layer)%twodarray, dim = 2)
                          temp(p) = 0.0d0
                          do q = 1, &
                          size(unraveled_variables(n_symbol)&
                          %weights(layer)%twodarray, dim = 1) - 1
                              temp(p) = temp(p) + &
                              der_coordinates_o(l, i)%onedarray(&
                              layer)%onedarray(q) * &
                              unraveled_variables(n_symbol)&
                              %weights(layer)%twodarray(q, p)
                          end do
                      end do
                      q = size(o(n_index)%onedarray(&
                      layer + 1)%onedarray)
                      allocate(der_coordinates_o(l, i)%&
                      onedarray(layer + 1)%onedarray(q))
                      do p = 1, &
                      size(o(n_index)%onedarray(layer + 1)%onedarray)
                          if (activation == 'tanh') then
                              der_coordinates_o(l, i)%onedarray(&
                              layer + 1)%onedarray(p) = temp(p) * &
                              (1.0 - o(n_index)%onedarray(layer + 1)&
                              %onedarray(p) * o(n_index)%onedarray(&
                              layer + 1)%onedarray(p))
                          else if (activation == 'sigmoid') then
                              der_coordinates_o(l, i)%onedarray(&
                              layer + 1)%onedarray(p) = temp(p) * &
                              (1.0 - o(n_index)%onedarray(layer + 1)%&
                              onedarray(p)) * o(n_index)%onedarray(&
                              layer + 1)%onedarray(p)
                          else if (activation == 'linear') then
                              der_coordinates_o(l, i)%onedarray(layer&
                               + 1)%onedarray(p) = temp(p)
                          end if
                      end do
                      deallocate(temp)
                  end do
                  allocate(der_coordinates_D(nn + 1))
                  do layer = 1, nn + 1
                      allocate(der_coordinates_D(layer)%onedarray(&
                      size(o(n_index)%onedarray(&
                      layer + 1)%onedarray)))
                      do m = 1, size(o(n_index)%onedarray(&
                      layer + 1)%onedarray)
                          if (activation == "tanh") then
                              der_coordinates_D(layer)%onedarray(m) =&
                              - 2.0d0 * o(n_index)%onedarray(&
                              layer + 1)%onedarray(m) * &
                              der_coordinates_o(l, i)%onedarray(&
                              layer + 1)%onedarray(m)
                          elseif (activation == "sigmoid") then
                              der_coordinates_D(layer)%onedarray(m) =&
                              der_coordinates_o(l, i)%onedarray(&
                              layer + 1)%onedarray(m) * &
                              ( 1.0d0 - 2.0d0 * o(n_index)%onedarray(&
                              layer + 1)%onedarray(m))
                          elseif (activation == "linear") then
                              der_coordinates_D(layer)%onedarray(m) =&
                              0.0d0
                          end if
                      end do    
                  end do
                  allocate(der_coordinates_delta(nn + 1))
                  allocate(der_coordinates_delta(nn + 1)%onedarray(1))
                  der_coordinates_delta(nn + 1)%onedarray(1) = &
                  der_coordinates_D(nn + 1)%onedarray(1)
                  do layer = nn, 1, -1
                      allocate(temp3(size(unraveled_variables(&
                      n_symbol)%weights(&
                      layer + 1)%twodarray, dim = 1) - 1))
                      allocate(temp4(size(unraveled_variables(&
                      n_symbol)%weights(&
                      layer + 1)%twodarray, dim = 1) - 1))
                      do p = 1, size(unraveled_variables(&
                      n_symbol)%weights(&
                      layer + 1)%twodarray, dim = 1) - 1
                          temp3(p) = 0.0d0
                          temp4(p) = 0.0d0
                          do q = 1, size(delta(n_index)%onedarray(&
                          layer + 1)%onedarray)
                              temp3(p) = temp3(p) + &
                              unraveled_variables(n_symbol)&
                              %weights(layer + 1)%twodarray(&
                              p, q) * delta(n_index)%onedarray(&
                              layer + 1)%onedarray(q)
                              temp4(p) = temp4(p) + &
                              unraveled_variables(n_symbol)&
                              %weights(layer + 1)%twodarray(&
                              p, q) * der_coordinates_delta(&
                              layer + 1)%onedarray(q)
                          end do
                      end do
                      allocate(temp5(&
                      size(der_coordinates_D(layer)%onedarray)))
                      allocate(temp6(size(&
                      der_coordinates_D(layer)%onedarray)))
                      allocate(der_coordinates_delta(&
                      layer)%onedarray(size(der_coordinates_D(&
                      layer)%onedarray)))
                       do p = 1, size(&
                      der_coordinates_D(layer)%onedarray)
                          temp5(p) = der_coordinates_D(layer)%&
                          onedarray(p) * temp3(p)
                          temp6(p) = D(n_index)%onedarray(&
                          layer)%onedarray(p) * temp4(p)
                          der_coordinates_delta(layer)%onedarray(p)= &
                          temp5(p) + temp6(p)
                      end do
                      deallocate(temp3)
                      deallocate(temp4)
                      deallocate(temp5)
                      deallocate(temp6)
                  end do
                  nn_forces(self_index, i) = &
                  nn_forces(self_index, i) - &
                  unraveled_variables(n_symbol)%scaling_slope*&
                  der_coordinates_o(l, i)%onedarray(nn + 2)%&
                  onedarray(1)
                  do p = 1, size(der_coordinates_delta)
                      deallocate(der_coordinates_delta(p)%onedarray)
                  end do
                  deallocate(der_coordinates_delta)
                  do p = 1, size(der_coordinates_D)
                      deallocate(der_coordinates_D(p)%onedarray)
                  end do
                  deallocate(der_coordinates_D)
                  do j = 1, size(der_coordinates_o(l, i)%onedarray)
                      deallocate(&
                      der_coordinates_o(l, i)%onedarray(j)%onedarray)
                  end do
                  deallocate(der_coordinates_o(l, i)%onedarray)
              end do  
          end do  
          deallocate(der_coordinates_o)
          deallocate(n_self_indices)
      end do
      
      do atom = 1, no_of_atoms
          do layer = 1, size(o(atom)%onedarray)
              deallocate(o(atom)%onedarray(layer)%onedarray)
          end do
          deallocate(o(atom)%onedarray)
      end do
      do atom = 1, no_of_atoms
          do layer = 1, size(ohat(atom)%onedarray)
              deallocate(ohat(atom)%onedarray(layer)%onedarray)
          end do
          deallocate(ohat(atom)%onedarray)
      end do

      do i = 1, size(delta)
        do j = 1, size(delta(i)%onedarray)
            deallocate(delta(i)%onedarray(j)%onedarray)
        end do
        deallocate(delta(i)%onedarray)
      end do

      do i = 1, size(D)
        do j = 1, size(D(i)%onedarray)
            deallocate(D(i)%onedarray(j)%onedarray)
        end do
        deallocate(D(i)%onedarray)
      end do

      end subroutine calculate_nn_forces
      
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine scale_fingerprints() 
      
      double precision :: double_temp
  
      do index = 1, size(unraveled_fingerprints)
          do symbol = 1, no_of_elements
              if (atomic_numbers_of_image(index)&             
               == elements_numbers(symbol)) then
                  exit
              end if
          end do    
          do l = 1, len_fingerprints_of_elements(symbol)
              if ((max_fingerprints(symbol, l) - &
              min_fingerprints(symbol, l)) .GT. (10.0d0 ** (-8.0d0)))&
               then
                  double_temp = &
                  unraveled_fingerprints(index)%onedarray(l)
                  double_temp = -1.0d0 + 2.0d0 * (double_temp - &
                  min_fingerprints(symbol, l)) / &
                  (max_fingerprints(symbol, l) - &
                  min_fingerprints(symbol, l))
                  unraveled_fingerprints(index)%onedarray(l) = &
                  double_temp
              endif
          end do
      end do

      end subroutine scale_fingerprints

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine scale_der_fingerprints() 
      
      integer :: n_symbol
      double precision :: double_temp
      integer, allocatable :: n_self_indices(:)

      do self_index = 1, size(unraveled_der_fingerprints)
          allocate(n_self_indices(size(unraveled_neighborlists(&
          self_index)%onedarray)))
          do p = 1, size(unraveled_neighborlists(self_index)%onedarray)
              n_self_indices(p) = unraveled_neighborlists(&
              self_index)%onedarray(p)
          end do
          do n_index = 1, size(n_self_indices)
              do n_symbol = 1, no_of_elements
              if (atomic_numbers_of_image(n_self_indices(n_index)) == &
              elements_numbers(n_symbol)) then
                  exit
              end if
              end do
              do p = 1, 3
                  do q = 1, len_fingerprints_of_elements(n_symbol)
                      if ((max_fingerprints(n_symbol, q) - &
                      min_fingerprints(n_symbol, q)) .GT. &
                      (10.0d0 ** (-8.0d0))) then
                          double_temp = &
                          unraveled_der_fingerprints(&
                          self_index)%onedarray(n_index)%twodarray(p, q)
                          double_temp = 2.0d0 * double_temp / &
                          (max_fingerprints(n_symbol, q) - &
                          min_fingerprints(n_symbol, q))
                          unraveled_der_fingerprints(&
                          self_index)%onedarray(&
                          n_index)%twodarray(p, q) = double_temp
                      endif
                  end do
              end do
          end do
          deallocate(n_self_indices)
      end do

      end subroutine scale_der_fingerprints
 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      end subroutine get_forces
 
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      module images_data

      double precision :: energy_coefficient
      double precision :: force_coefficient
      integer :: no_of_elements, no_of_images
      integer, allocatable :: elements_numbers(:)
      integer :: len_of_no_nodes_of_elements
      integer, allocatable :: no_layers_of_elements(:)
      integer, allocatable :: no_nodes_of_elements(:)
      double precision, allocatable :: real_energies(:)
      double precision, allocatable :: real_forces(:, :)
      integer, allocatable :: no_of_atoms_of_images(:)
      integer, allocatable :: atomic_numbers_of_images(:)
      double precision, allocatable :: raveled_fps_of_images(:, :)
      integer, allocatable :: len_fingerprints_of_elements(:)
      logical :: train_forces
      integer, allocatable :: list_of_no_of_neighbors(:)
      integer, allocatable :: raveled_neighborlists(:)
      double precision, allocatable :: raveled_der_fingerprints(:, :)
      integer :: no_procs
      integer, allocatable :: no_sub_images(:)
      double precision, allocatable ::min_fingerprints(:, :)
      double precision, allocatable ::max_fingerprints(:, :)

      end module images_data

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine share_cost_function_task_between_cores(&
        proc, len_of_variables, variables, &
        activation, energy_square_error, &
        force_square_error, der_variables_square_error)
                        
      use images_data                        
      implicit none

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! input/output variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      character(len=7) :: activation
      integer :: len_of_variables, proc
      double precision :: variables(len_of_variables)
      double precision :: energy_square_error, force_square_error
      double precision :: der_variables_square_error(len_of_variables)
!f2py         intent(in) :: variables , len_of_variables, activation, proc
!f2py         intent(out) :: energy_square_error, force_square_error
!f2py         intent(out) :: der_variables_square_error
!f2py         intent(hidden) :: o, D, ohat, nn_energy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! type definition !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type :: real_one_d_array_type
        sequence
        double precision, allocatable:: onedarray(:)
      end type real_one_d_array_type

      type :: integer_one_d_array_type
        sequence
        integer, allocatable:: onedarray(:)
      end type integer_one_d_array_type

      type :: real_two_d_array_type
        sequence
        double precision, allocatable:: twodarray(:,:)
      end type real_two_d_array_type

      type :: integer_two_d_array_type
        sequence
        integer, allocatable:: twodarray(:,:)
      end type integer_two_d_array_type

      type :: image_forces_type
        sequence
        double precision, allocatable:: atomic_forces(:, :)
      end type image_forces_type

      type :: embedded_real_one_two_d_array_type
        sequence
        type(real_two_d_array_type), allocatable:: onedarray(:)
      end type embedded_real_one_two_d_array_type

      type :: embedded_integer_one_two_d_array_type
        sequence
        type(integer_two_d_array_type), allocatable:: onedarray(:)
      end type embedded_integer_one_two_d_array_type

      type :: embedded_real_one_one_d_array_type
        sequence
        type(real_one_d_array_type), allocatable:: onedarray(:)
      end type embedded_real_one_one_d_array_type

      type :: embedded_integer_one_one_d_array_type
        sequence
        type(integer_one_d_array_type), allocatable:: onedarray(:)
      end type embedded_integer_one_one_d_array_type

      type :: embedded_one_one_two_d_array_type
        sequence
        type(embedded_real_one_two_d_array_type), allocatable:: &
        onedarray(:)
      end type embedded_one_one_two_d_array_type

      type :: element_variables_type
        sequence
        double precision :: scaling_intercept
        double precision :: scaling_slope
        type(real_two_d_array_type), allocatable:: weights(:)
      end type element_variables_type

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! dummy variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type(real_one_d_array_type), allocatable :: &
      unraveled_fingerprints(:)
      type(embedded_real_one_one_d_array_type) :: &
      unraveled_fingerprints_of_images(no_sub_images(proc))
      type(integer_one_d_array_type) :: &
      unraveled_atomic_numbers(no_sub_images(proc))
      type(element_variables_type) :: &
      unraveled_variables(no_of_elements)
      type(element_variables_type) :: &
      unraveled_der_variables(no_of_elements)
      type(embedded_real_one_one_d_array_type), allocatable :: &
      o(:), ohat(:)
      double precision :: nn_energy
      integer :: i, index, j, m, n, p, layer, nn, k, q, l, no_of_rows, &
      no_of_cols, symbol, image_no, no_of_atoms, element, atom
      double precision :: &
      image_der_variables_square_error(len_of_variables)
      type(image_forces_type) :: &
      unraveled_real_forces(no_sub_images(proc))
      type(embedded_integer_one_one_d_array_type) :: &
      unraveled_neighborlists(no_sub_images(proc))
      type(embedded_one_one_two_d_array_type) :: &
      unraveled_der_fingerprints_of_images(no_sub_images(proc))
      integer :: n_index, self_index
      integer :: first_image_no, no_proc_images

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! calculations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      first_image_no = 1
      do i = 1, proc - 1
        first_image_no = first_image_no + no_sub_images(i)
      end do
      no_proc_images = no_sub_images(proc)
      energy_square_error = 0.0d0
      force_square_error = 0.0d0
      do i = 1, len_of_variables
        der_variables_square_error(i) = 0.0d0
      end do
      call unravel_variables()
      call unravel_atomic_numbers(first_image_no, no_proc_images)
      call unravel_fingerprints(first_image_no, no_proc_images)
      call scale_fingerprints(no_proc_images)
      if (train_forces .eqv. .true.) then
        call unravel_neighborlists(first_image_no, no_proc_images)
        call unravel_der_fingerprints(first_image_no, no_proc_images)
        call scale_der_fingerprints(no_proc_images)
        call unravel_real_forces(first_image_no, no_proc_images)
      end if

! summation over images
      do image_no = 1, no_proc_images
        do element = 1, len_of_variables
            image_der_variables_square_error(element) = 0.0d0
        end do
        no_of_atoms = no_of_atoms_of_images(&
        first_image_no - 1 + image_no)
        allocate(unraveled_fingerprints(no_of_atoms))
        do p = 1, size(&
        unraveled_fingerprints_of_images(image_no)%onedarray)
            allocate(unraveled_fingerprints(p)%onedarray(&
            size(unraveled_fingerprints_of_images(&
            image_no)%onedarray(p)%onedarray)))
            do q = 1, size(unraveled_fingerprints_of_images(&
            image_no)%onedarray(p)%onedarray)
                unraveled_fingerprints(p)%onedarray(q) =&
                unraveled_fingerprints_of_images(image_no)&
                %onedarray(p)%onedarray(q)
            end do 
        end do
        nn_energy = calculate_nn_energy(image_no, no_of_atoms, &
        unraveled_fingerprints, unraveled_variables)
        energy_square_error = energy_square_error + &
        (nn_energy - &
        real_energies(first_image_no - 1 + image_no)) ** 2.0d0 / &
        (no_of_atoms ** 2.0d0)
        call calculate_der_variables(first_image_no, image_no, &
        no_of_atoms, unraveled_variables, nn_energy)
        image_der_variables_square_error = &
        ravel_variables(unraveled_der_variables)
        do i = 1, len_of_variables
            der_variables_square_error(i) = &
            der_variables_square_error(i) &
            + image_der_variables_square_error(i)
        end do
        
        !deallocations for each image
        do atom = 1, no_of_atoms
            deallocate(unraveled_fingerprints(atom)%onedarray)
        end do
        deallocate(unraveled_fingerprints)
        do element = 1, no_of_elements
            do layer = 1, size(&
            unraveled_der_variables(element)%weights)
                deallocate(unraveled_der_variables(element)%weights(&
                layer)%twodarray)
            end do
            deallocate(unraveled_der_variables(element)%weights)
        end do
        do atom = 1, no_of_atoms
            do layer = 1, size(o(atom)%onedarray)
                deallocate(o(atom)%onedarray(layer)%onedarray)
            end do
            deallocate(o(atom)%onedarray)
        end do
        deallocate(o)
        do atom = 1, no_of_atoms
            do layer = 1, size(ohat(atom)%onedarray)
                deallocate(ohat(atom)%onedarray(layer)%onedarray)
            end do
            deallocate(ohat(atom)%onedarray)
        end do
        deallocate(ohat)
      end do

!deallocations for all images
      do element = 1, size(unraveled_variables)
        do layer = 1, size(unraveled_variables(element)%weights)
            deallocate(&
            unraveled_variables(element)%weights(layer)%twodarray)
        end do
        deallocate(unraveled_variables(element)%weights)
      end do
      do image_no = 1, no_proc_images
        deallocate(unraveled_atomic_numbers(image_no)%onedarray)
      end do
      do image_no = 1, no_proc_images
        no_of_atoms = &
        no_of_atoms_of_images(first_image_no - 1 + image_no)
        do index = 1, no_of_atoms
            deallocate(unraveled_fingerprints_of_images(&
            image_no)%onedarray(index)%onedarray)
        end do
        deallocate(unraveled_fingerprints_of_images(image_no)%onedarray)
      end do
      if (train_forces .eqv. .true.) then
        do image_no = 1, no_proc_images
            no_of_atoms = &
            no_of_atoms_of_images(first_image_no - 1 + image_no)
            do self_index = 1, no_of_atoms
                do n_index = 1, &
                    size(unraveled_der_fingerprints_of_images(&
                    image_no)%onedarray(self_index)%onedarray)
                    deallocate(unraveled_der_fingerprints_of_images(&
                    image_no)%onedarray(self_index)%onedarray(&
                    n_index)%twodarray)
                end do
                deallocate(unraveled_der_fingerprints_of_images(&
                image_no)%onedarray(self_index)%onedarray)
            end do
            deallocate(unraveled_der_fingerprints_of_images(&
            image_no)%onedarray)
        end do
        do image_no = 1, no_proc_images
            deallocate(unraveled_real_forces(image_no)%atomic_forces)
        end do
        do image_no = 1, no_proc_images
            no_of_atoms = &
            no_of_atoms_of_images(first_image_no - 1 + image_no)
            do index = 1, no_of_atoms
                deallocate(unraveled_neighborlists(&
                image_no)%onedarray(index)%onedarray)
            end do
            deallocate(unraveled_neighborlists(image_no)%onedarray)
        end do
      end if

      contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine unravel_variables()
 
      k = 0
      l = 0
      do symbol = 1, no_of_elements
        allocate(unraveled_variables(symbol)%weights(&
        no_layers_of_elements(symbol)-1))
        if (symbol .GT. 1) then
            k = k + no_layers_of_elements(symbol - 1)
        end if
        do j = 1, no_layers_of_elements(symbol) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_variables(symbol)%weights(j)%twodarray(&
            no_of_rows, no_of_cols))
            do m = 1, no_of_rows
                do n = 1, no_of_cols
                    unraveled_variables(symbol)%weights(j)%twodarray(&
                    m, n) = variables(l + (m - 1) * no_of_cols + n)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do symbol = 1, no_of_elements
        unraveled_variables(symbol)%scaling_intercept = &
        variables(l + 2 *  symbol - 1)
        unraveled_variables(symbol)%scaling_slope = &
        variables(l + 2 * symbol)
      end do
      end subroutine unravel_variables

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_der_fingerprints(first_image_no, &
      no_proc_images)

      integer :: first_image_no, no_proc_images
      integer :: no_of_neighbors, n_symbol, temp
      integer, allocatable :: n_self_indices(:)

      k = 0
      temp = 0
      if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            no_of_atoms = no_of_atoms_of_images(image_no)
            do index = 1, no_of_atoms
                temp = temp + list_of_no_of_neighbors(k + index)
            end do
            k = k + no_of_atoms
         end do
      end if
      do image_no = 1, no_proc_images
        no_of_atoms = &
        no_of_atoms_of_images(first_image_no - 1 + image_no)
        allocate(unraveled_der_fingerprints_of_images(&
        image_no)%onedarray(no_of_atoms))
        do self_index = 1, no_of_atoms
            allocate(n_self_indices(size(unraveled_neighborlists(&
            image_no)%onedarray(self_index)%onedarray)))
            do p = 1, size(unraveled_neighborlists(image_no)%&
            onedarray(self_index)%onedarray)
                n_self_indices(p) = unraveled_neighborlists(image_no)%&
                onedarray(self_index)%onedarray(p)
            end do
            no_of_neighbors = list_of_no_of_neighbors(k + self_index)
            allocate(unraveled_der_fingerprints_of_images(&
            image_no)%onedarray(self_index)%onedarray(no_of_neighbors))
            do n_index = 1, no_of_neighbors
                do n_symbol = 1, no_of_elements
                if (unraveled_atomic_numbers(&
                image_no)%onedarray(n_self_indices(n_index)) == &
                elements_numbers(n_symbol)) then
                    exit
                end if
                end do
                allocate(unraveled_der_fingerprints_of_images(&
                image_no)%onedarray(&
                self_index)%onedarray(n_index)%twodarray&
                (3, len_fingerprints_of_elements(n_symbol)))
                do p = 1, 3
                    do q = 1, len_fingerprints_of_elements(n_symbol)
                        unraveled_der_fingerprints_of_images(&
                        image_no)%onedarray(&
                        self_index)%onedarray(n_index)%twodarray(p, q) &
                        = raveled_der_fingerprints(&
                        3 * temp + 3 * n_index + p - 3, q)
                    end do
                end do
            end do
            deallocate(n_self_indices)
            temp = temp + no_of_neighbors
        end do
        k = k + no_of_atoms
      end do
      end subroutine unravel_der_fingerprints

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 
      subroutine unravel_real_forces(first_image_no, no_proc_images)

      integer :: first_image_no, no_proc_images

      k = 0
       if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            no_of_atoms = no_of_atoms_of_images(image_no)
            k = k + no_of_atoms
        end do
      end if
      do image_no = 1, no_proc_images
        no_of_atoms = &
        no_of_atoms_of_images(first_image_no - 1 + image_no)
        allocate(unraveled_real_forces(image_no)%atomic_forces(&
        no_of_atoms, 3))
        do atom = 1, no_of_atoms
            do i = 1, 3
                unraveled_real_forces(image_no)%atomic_forces(&
                atom, i) = real_forces(k + atom, i)
            end do
        end do
        k = k + no_of_atoms
      end do
      end subroutine unravel_real_forces

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine unravel_neighborlists(first_image_no, no_proc_images)

      integer :: first_image_no, no_proc_images
      integer :: temp

      k = 0
      temp = 0
      if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            no_of_atoms = no_of_atoms_of_images(image_no)
            do index = 1, no_of_atoms
                temp = temp + list_of_no_of_neighbors(k + index)
            end do
            k = k + no_of_atoms
        end do
      end if
      do image_no = 1, no_proc_images
        no_of_atoms = &
        no_of_atoms_of_images(first_image_no - 1 + image_no)
        allocate(unraveled_neighborlists(&
        image_no)%onedarray(no_of_atoms))
        do index = 1, no_of_atoms
            allocate(unraveled_neighborlists(image_no)%onedarray(&
            index)%onedarray(list_of_no_of_neighbors(k + index)))
            do p = 1, list_of_no_of_neighbors(k + index)
                unraveled_neighborlists(image_no)%onedarray(&
                index)%onedarray(p) = raveled_neighborlists(temp + p)+1
            end do
            temp = temp + list_of_no_of_neighbors(k + index)
        end do
        k = k + no_of_atoms
      end do
      end subroutine unravel_neighborlists

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine unravel_atomic_numbers(first_image_no, no_proc_images)
 
      integer :: first_image_no, no_proc_images

      k = 0
      if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            no_of_atoms = no_of_atoms_of_images(image_no)
            k = k + no_of_atoms
        end do
      end if
      do image_no = 1, no_proc_images
        no_of_atoms = &
        no_of_atoms_of_images(first_image_no - 1 + image_no)
        allocate(unraveled_atomic_numbers(image_no)%onedarray(&
        no_of_atoms))
        do l = 1, no_of_atoms
            unraveled_atomic_numbers(image_no)%onedarray(l) = &
            atomic_numbers_of_images(k + l)
        end do
        k = k + no_of_atoms
      end do
      end subroutine unravel_atomic_numbers
 
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_fingerprints(first_image_no, no_proc_images) 
      
      integer :: first_image_no, no_proc_images

      k = 0
      if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            no_of_atoms = no_of_atoms_of_images(image_no)
            k = k + no_of_atoms
        end do
      end if
      do image_no = 1, no_proc_images
        no_of_atoms = &
        no_of_atoms_of_images(first_image_no - 1 + image_no)
        allocate(unraveled_fingerprints_of_images(&
        image_no)%onedarray(no_of_atoms))
        do index = 1, no_of_atoms
            do symbol = 1, no_of_elements
                if (unraveled_atomic_numbers(image_no)%onedarray(index)&
                 == elements_numbers(symbol)) then
                    allocate(unraveled_fingerprints_of_images(&
                    image_no)%onedarray(index)%onedarray(&
                    len_fingerprints_of_elements(symbol))) 
                    exit
                end if
            end do    
            do l = 1, len_fingerprints_of_elements(symbol)
                unraveled_fingerprints_of_images(&
                image_no)%onedarray(index)&
                %onedarray(l) = raveled_fps_of_images(k + index, l)
            end do
        end do
      k = k + no_of_atoms
      end do
      end subroutine unravel_fingerprints


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      function calculate_nn_energy(image_no, no_of_atoms, &
      unraveled_fingerprints, unraveled_variables)
 
      integer :: image_no, no_of_atoms
      type(real_one_d_array_type) :: unraveled_fingerprints(no_of_atoms)
      type(element_variables_type) :: &
      unraveled_variables(no_of_elements)
      double precision :: calculate_nn_energy

      double precision :: atomic_nn_energy
      integer, allocatable :: hiddensizes(:)
      type(real_one_d_array_type), allocatable :: atomic_net(:)
 
      calculate_nn_energy = 0.0d0
      allocate(o(no_of_atoms))
      allocate(ohat(no_of_atoms))
      do index = 1, no_of_atoms
        p = 0
        do symbol = 1, no_of_elements
            if (unraveled_atomic_numbers(image_no)%onedarray(index)== &
            elements_numbers(symbol)) then
                exit
            else 
                p = p + no_layers_of_elements(symbol)
            end if
        end do
        allocate(hiddensizes(no_layers_of_elements(symbol) - 2))
        do m = 1, no_layers_of_elements(symbol) - 2
            hiddensizes(m) = no_nodes_of_elements(p + m + 1)
        end do
        allocate(o(index)%onedarray(no_layers_of_elements(symbol)))
        allocate(ohat(index)%onedarray(no_layers_of_elements(symbol)))
        allocate(atomic_net(no_layers_of_elements(symbol)))
        layer = 1
        allocate(o(index)%onedarray(1)%onedarray(size(&
        unraveled_fingerprints(index)%onedarray)))
        allocate(ohat(index)%onedarray(1)%onedarray(&
        size(unraveled_fingerprints(index)%onedarray) + 1))
        do m = 1, size(unraveled_variables(symbol)&
        %weights(1)%twodarray, dim=1) - 1
            o(index)%onedarray(1)%onedarray(m) = &
            unraveled_fingerprints(index)%onedarray(m)
        end do
        do layer = 1, size(hiddensizes) + 1
            do m = 1, size(unraveled_variables(symbol)&
            %weights(layer)%twodarray, dim=1) - 1
                ohat(index)%onedarray(layer)%onedarray(m) = &
                o(index)%onedarray(layer)%onedarray(m)
            end do
            ohat(index)%onedarray(layer)%onedarray(&
            size(unraveled_variables(symbol)&
            %weights(layer)%twodarray, dim=1)) = 1.0d0
            allocate(atomic_net(layer)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)))
            allocate(o(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)))
            allocate(ohat(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2) + 1))
            do m = 1, size(unraveled_variables(symbol)%&
            weights(layer)%twodarray, dim=2)
                atomic_net(layer)%onedarray(m) = 0.0d0
                do n = 1, size(unraveled_variables(symbol)%&
                weights(layer)%twodarray, dim=1)
                    atomic_net(layer)%onedarray(m) =  &
                    atomic_net(layer)%onedarray(m) + &
                    ohat(index)%onedarray(layer)%onedarray(n) * &
                    unraveled_variables(symbol)&
                    %weights(layer)%twodarray(n, m)
                end do
                if (activation == 'tanh') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    tanh(atomic_net(layer)%onedarray(m))
                else if (activation == 'sigmoid') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    1. / (1. +  exp(- atomic_net(layer)%onedarray(m)))
                else if (activation == 'linear') then
                    o(index)%onedarray(layer + 1)%onedarray(m) = &
                    atomic_net(layer)%onedarray(m)
                else
                    print *, &
                    "Fortran Error: Unknown activation function!"
                end if
                ohat(index)%onedarray(layer + 1)%onedarray(m) = &
                o(index)%onedarray(layer + 1)%onedarray(m)
            end do
            ohat(index)%onedarray(layer + 1)%onedarray(&
            size(unraveled_variables(symbol)%weights(&
            layer)%twodarray, dim=2) + 1) =  1.0
        end do
        atomic_nn_energy = unraveled_variables(symbol)&
        %scaling_slope * o(index)%onedarray(&
        layer)%onedarray(1) + unraveled_variables(&
        symbol)%scaling_intercept
        calculate_nn_energy = calculate_nn_energy +  atomic_nn_energy
        deallocate(hiddensizes)
        deallocate(atomic_net)
        end do

      end function calculate_nn_energy
 
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine calculate_der_variables(first_image_no, image_no, &
      no_of_atoms, unraveled_variables, nn_energy)
 
      integer :: first_image_no, image_no, no_of_atoms
      type(element_variables_type) :: &
      unraveled_variables(no_of_elements)
      double precision :: nn_energy, temp1, temp2
 
      type(embedded_real_one_one_d_array_type) :: delta(no_of_atoms)
      type(embedded_real_one_one_d_array_type) :: D(no_of_atoms)

      type (embedded_real_one_one_d_array_type), allocatable :: &
      der_coordinates_o(:, :)
      type (embedded_real_one_two_d_array_type), allocatable :: &
      der_coordinates_weights_atomic_output(:, :)
      type(real_one_d_array_type), allocatable :: &
      der_coordinates_ohat(:)
      integer, allocatable :: n_self_indices(:), n_symbols(:)
      integer :: n_symbol, n_index
      double precision, allocatable :: temp(:)
      type(real_one_d_array_type), allocatable :: der_coordinates_D(:)
      type (real_one_d_array_type), allocatable :: &
      der_coordinates_delta(:)
      double precision :: nn_atomic_forces(no_of_atoms, 3)
      double precision, allocatable :: temp3(:), temp4(:), temp5(:), &
      temp6(:)
 
      do symbol = 1, no_of_elements
        unraveled_der_variables(symbol)%scaling_intercept = 0.d0
        unraveled_der_variables(symbol)%scaling_slope = 0.d0
      end do
 
      k = 0
      l = 0
      do symbol = 1, no_of_elements
        allocate(unraveled_der_variables(symbol)%weights(&
        no_layers_of_elements(symbol)-1))
        if (symbol > 1) then
            k = k + no_layers_of_elements(symbol - 1)
        end if
        do j = 1, no_layers_of_elements(symbol) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_der_variables(symbol)%weights(j)&
            %twodarray(no_of_rows, no_of_cols))
            do m = 1, no_of_rows
                do n = 1, no_of_cols
                    unraveled_der_variables(symbol)%weights(&
                    j)%twodarray(m, n) = 0.0d0
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
 
      do index = 1, no_of_atoms   
        do symbol = 1, no_of_elements
            if (unraveled_atomic_numbers(image_no)%onedarray(index) == &
            elements_numbers(symbol)) then
                exit
            end if
        end do
        nn = size(o(index)%onedarray) - 2
        allocate(D(index)%onedarray(nn + 1))
        do layer = 1, nn + 1
            allocate(D(index)%onedarray(layer)%onedarray(&
            size(o(index)%onedarray(layer + 1)%onedarray)))
            do m = 1, size(o(index)%onedarray(layer + 1)%onedarray)
                if (activation == "tanh") then
                    D(index)%onedarray(layer)%onedarray(m) = &
                    (1.0d0 - o(index)%onedarray(layer + 1)%onedarray(m)&
                     * o(index)%onedarray(layer + 1)%onedarray(m))
                elseif (activation == "sigmoid") then
                    D(index)%onedarray(layer)%onedarray(m) = &
                    o(index)%onedarray(layer + 1)%onedarray(m) * &
                    (1.0d0 - o(index)%onedarray(layer + 1)%onedarray(m))
                elseif (activation == "linear") then
                    D(index)%onedarray(layer)%onedarray(m) = 1.0d0
                end if
            end do    
        end do
        allocate(delta(index)%onedarray(nn + 1))
        allocate(delta(index)%onedarray(nn + 1)%onedarray(1))
        delta(index)%onedarray(nn + 1)%onedarray(1) = &
        D(index)%onedarray(nn + 1)%onedarray(1)
        do layer = nn, 1, -1
            allocate(delta(index)%onedarray(layer)%onedarray(&
            size(D(index)%onedarray(layer)%onedarray)))
            do m = 1, size(D(index)%onedarray(layer)%onedarray)
                delta(index)%onedarray(layer)%onedarray(m) = 0.0d0
                do q = 1, &
                size(delta(index)%onedarray(layer + 1)%onedarray)
                    temp1 = D(index)%onedarray(layer)%onedarray(m) * &
                    unraveled_variables(symbol)%weights(layer + 1)&
                    %twodarray(m, q)
                    temp2 = temp1  * &
                    delta(index)%onedarray(layer + 1)%onedarray(q)
                    delta(index)%onedarray(layer)%onedarray(m) = &
                    delta(index)%onedarray(layer)%onedarray(m) + temp2
                end do
            end do
        end do
    
        unraveled_der_variables(symbol)%scaling_intercept = &
        unraveled_der_variables(symbol)%scaling_intercept + &
        energy_coefficient *  2. * (nn_energy - real_energies(&
        first_image_no - 1 + image_no)) / (no_of_atoms ** 2.0d0)
        unraveled_der_variables(symbol)%scaling_slope = &
        unraveled_der_variables(symbol)%scaling_slope + &
        energy_coefficient *  2.0d0 * &
        (nn_energy - real_energies(first_image_no - 1 + image_no)) * &
        o(index)%onedarray(nn + 2)%onedarray(1) &
         / (no_of_atoms ** 2.0d0)
        do layer = 1, nn + 1
            do m = 1, size(ohat(index)%onedarray(layer)%onedarray)
                do n = 1, size(delta(index)%onedarray(layer)%onedarray)
                    unraveled_der_variables(symbol)%weights(&
                    layer)%twodarray(m, n) = &
                    unraveled_der_variables(&
                    symbol)%weights(layer)%twodarray(m, n) &
                    + energy_coefficient *  2. * &
                    (nn_energy - real_energies(&
                    first_image_no - 1 + image_no)) * &
                    unraveled_variables(symbol)%scaling_slope &
                    * ohat(index)%onedarray(layer)%onedarray(m) &
                    * delta(index)%onedarray(layer)%onedarray(n) &
                    / (no_of_atoms ** 2.0d0)
                end do
            end do
      end do
      end do
 
      if (train_forces .eqv. .true.) then
        do self_index = 1, no_of_atoms
            do i = 1, 3
                nn_atomic_forces(self_index, i) = 0.0d0
            end do
        end do
        do self_index = 1, no_of_atoms
            allocate(n_self_indices(size(unraveled_neighborlists(&
            image_no)%onedarray(self_index)%onedarray)))
            do p = 1, size(unraveled_neighborlists(image_no)%&
            onedarray(self_index)%onedarray)
                n_self_indices(p) = unraveled_neighborlists(image_no)%&
                onedarray(self_index)%onedarray(p)
            end do 
            allocate(n_symbols(size(n_self_indices)))
            allocate(der_coordinates_weights_atomic_output(&
            size(n_self_indices), 3))
            allocate(der_coordinates_o(size(n_self_indices), 3))
            do l = 1, size(n_self_indices)
                n_index = n_self_indices(l)
                do n_symbol = 1, no_of_elements
                    if (unraveled_atomic_numbers(image_no)%onedarray(&
                    n_index) == elements_numbers(n_symbol)) then
                        exit
                    end if
                end do
                nn = size(o(n_index)%onedarray) - 2
                do i = 1, 3
                    allocate(der_coordinates_o(l, i)%onedarray(nn + 2))
                    allocate(der_coordinates_o(l, i)%onedarray(1)%&
                    onedarray(len_fingerprints_of_elements(n_symbol)))
                    do m = 1, size(unraveled_der_fingerprints_of_images&
                    (image_no)%onedarray&
                    (self_index)%onedarray(l)%twodarray, dim = 2)
                    der_coordinates_o(l, i)%onedarray(1)%onedarray(m)=&
                    unraveled_der_fingerprints_of_images(image_no)&
                    %onedarray(self_index)%onedarray(l)%twodarray(i, m)
                    end do
                    do layer = 1, nn + 1
                        allocate(temp(size(unraveled_variables(&
                        n_symbol)%weights(layer)%twodarray, dim = 2)))
                        do p = 1, &
                        size(unraveled_variables(n_symbol)&
                        %weights(layer)%twodarray, dim = 2)
                            temp(p) = 0.0d0
                            do q = 1, &
                            size(unraveled_variables(n_symbol)&
                            %weights(layer)%twodarray, &
                            dim = 1) - 1
                                temp(p) = temp(p) + &
                                der_coordinates_o(l, i)%onedarray(&
                                layer)%onedarray(q) * &
                                unraveled_variables(n_symbol)&
                                %weights(layer)%twodarray(q, p)
                            end do
                        end do
                        q = size(o(n_index)%onedarray(&
                        layer + 1)%onedarray)
                        allocate(der_coordinates_o(l, i)%&
                        onedarray(layer + 1)%onedarray(q))
                        do p = 1, &
                        size(o(n_index)%onedarray(layer + 1)%onedarray)
                            if (activation == 'tanh') then
                                der_coordinates_o(l, i)%onedarray(&
                                layer + 1)%onedarray(p) = temp(p) * &
                                (1.0 - o(n_index)%onedarray(layer + 1)&
                                %onedarray(p) * o(n_index)%onedarray(&
                                layer + 1)%onedarray(p))
                            else if (activation == 'sigmoid') then
                                der_coordinates_o(l, i)%onedarray(&
                                layer + 1)%onedarray(p) = temp(p) * &
                                (1.0 - o(n_index)%onedarray(layer + 1)%&
                                onedarray(p)) * o(n_index)%onedarray(&
                                layer + 1)%onedarray(p)
                            else if (activation == 'linear') then
                                der_coordinates_o(l, i)%onedarray(layer&
                                 + 1)%onedarray(p) = temp(p)
                            end if
                        end do
                        deallocate(temp)
                    end do
                    allocate(der_coordinates_D(nn + 1))
                    do layer = 1, nn + 1
                        allocate(der_coordinates_D(layer)%onedarray(&
                        size(o(n_index)%onedarray(&
                        layer + 1)%onedarray)))
                        do m = 1, size(o(n_index)%onedarray(&
                        layer + 1)%onedarray)
                            if (activation == "tanh") then
                                der_coordinates_D(layer)%onedarray(m) =&
                                - 2.0d0 * o(n_index)%onedarray(&
                                layer + 1)%onedarray(m) * &
                                der_coordinates_o(l, i)%onedarray(&
                                layer + 1)%onedarray(m)
                            elseif (activation == "sigmoid") then
                                der_coordinates_D(layer)%onedarray(m) =&
                                der_coordinates_o(l, i)%onedarray(&
                                layer + 1)%onedarray(m) * &
                                ( 1.0d0 - 2.0d0 * o(n_index)%onedarray(&
                                layer + 1)%onedarray(m))
                            elseif (activation == "linear") then
                                der_coordinates_D(layer)%onedarray(m) =&
                                0.0d0
                            end if
                        end do    
                    end do
                    allocate(der_coordinates_delta(nn + 1))
                    allocate(der_coordinates_delta(nn + 1)%onedarray(1))
                    der_coordinates_delta(nn + 1)%onedarray(1) = &
                    der_coordinates_D(nn + 1)%onedarray(1)
                    do layer = nn, 1, -1
                        allocate(temp3(size(unraveled_variables(&
                        n_symbol)%weights(&
                        layer + 1)%twodarray, dim = 1) - 1))
                        allocate(temp4(size(unraveled_variables(&
                        n_symbol)%weights(&
                        layer + 1)%twodarray, dim = 1) - 1))
                        do p = 1, size(unraveled_variables(&
                        n_symbol)%weights(&
                        layer + 1)%twodarray, dim = 1) - 1
                            temp3(p) = 0.0d0
                            temp4(p) = 0.0d0
                            do q = 1, size(delta(n_index)%onedarray(&
                            layer + 1)%onedarray)
                                temp3(p) = temp3(p) + &
                                unraveled_variables(n_symbol)&
                                %weights(layer + 1)%twodarray(&
                                p, q) * delta(n_index)%onedarray(&
                                layer + 1)%onedarray(q)
                                temp4(p) = temp4(p) + &
                                unraveled_variables(n_symbol)&
                                %weights(layer + 1)%twodarray(&
                                p, q) * der_coordinates_delta(&
                                layer + 1)%onedarray(q)
                            end do
                        end do
                        allocate(temp5(&
                        size(der_coordinates_D(layer)%onedarray)))
                        allocate(temp6(size(&
                        der_coordinates_D(layer)%onedarray)))
                        allocate(der_coordinates_delta(&
                        layer)%onedarray(size(der_coordinates_D(&
                        layer)%onedarray)))
                        do p = 1, size(&
                        der_coordinates_D(layer)%onedarray)
                            temp5(p) = der_coordinates_D(layer)%&
                            onedarray(p) * temp3(p)
                            temp6(p) = D(n_index)%onedarray(&
                            layer)%onedarray(p) * temp4(p)
                            der_coordinates_delta(layer)%onedarray(p)= &
                            temp5(p) + temp6(p)
                        end do
                        deallocate(temp3)
                        deallocate(temp4)
                        deallocate(temp5)
                        deallocate(temp6)
                    end do
                    allocate(der_coordinates_ohat(nn + 1))
                    allocate(der_coordinates_weights_atomic_output(&
                    l, i)%onedarray(nn + 1))
                    do layer = 1, nn + 1
                        allocate(der_coordinates_ohat(layer)%onedarray(&
                        size(der_coordinates_o(l, i)%onedarray(layer)%&
                        onedarray) + 1))
                        do p = 1, size(der_coordinates_o(l, i)%&
                        onedarray(layer)%onedarray)
                            der_coordinates_ohat(layer)%onedarray(p) = &
                            der_coordinates_o(l, i)%onedarray(&
                            layer)%onedarray(p)
                        end do
                        der_coordinates_ohat(layer)%onedarray(&
                        size(der_coordinates_o(l, i)%onedarray(&
                        layer)%onedarray) + 1) = 0.0d0
                        allocate(der_coordinates_weights_atomic_output(&
                        l, i)%onedarray(layer)%twodarray &
                        (size(der_coordinates_ohat(layer)%onedarray), &
                        size(&
                        delta(n_index)%onedarray(layer)%onedarray)))
                        do p = 1, size(&
                        der_coordinates_ohat(layer)%onedarray)
                            do q = 1, size(delta(n_index)%onedarray(&
                            layer)%onedarray)
                                der_coordinates_weights_atomic_output&
                                (l, i)%onedarray(layer)%twodarray(p, q)&
                                = 0.0d0
                            end do
                        end do
                        do p = 1, size(&
                        der_coordinates_ohat(layer)%onedarray)
                            do q = 1, size(&
                            delta(n_index)%onedarray(layer)%onedarray)
                            der_coordinates_weights_atomic_output(l, i)&
                            %onedarray(layer)%twodarray(p, q) = &
                            der_coordinates_weights_atomic_output(l, i)&
                            %onedarray(layer)%twodarray(p, q) + &
                            der_coordinates_ohat(layer)%onedarray(p) * &
                            delta(n_index)%onedarray(layer)%&
                            onedarray(q) + &
                            ohat(n_index)%onedarray(layer)%onedarray(p)&
                            * der_coordinates_delta(layer)%onedarray(q)
                            end do
                        end do 
                    end do
                    nn_atomic_forces(self_index, i) = &
                    nn_atomic_forces(self_index, i) - &
                    unraveled_variables(n_symbol)%scaling_slope*&
                    der_coordinates_o(l, i)%onedarray(nn + 2)%&
                    onedarray(1)
                    do p = 1, size(der_coordinates_ohat)
                        deallocate(der_coordinates_ohat(p)%onedarray)
                    end do
                    deallocate(der_coordinates_ohat)
                    do p = 1, size(der_coordinates_delta)
                        deallocate(der_coordinates_delta(p)%onedarray)
                    end do
                    deallocate(der_coordinates_delta)
                    do p = 1, size(der_coordinates_D)
                        deallocate(der_coordinates_D(p)%onedarray)
                    end do
                    deallocate(der_coordinates_D)
                end do  
            end do  
            do i = 1, 3
                force_square_error = force_square_error + &
                (1.0d0 / 3.0d0)*(nn_atomic_forces(self_index, i)  - &
                unraveled_real_forces(image_no)%&
                atomic_forces(self_index, i)) ** 2.0 / no_of_atoms
                do l = 1, size(n_self_indices)
                    n_index = n_self_indices(l)
                    do n_symbol = 1, no_of_elements
                        if (unraveled_atomic_numbers(image_no)%&
                        onedarray(n_index) ==&
                         elements_numbers(n_symbol)) then
                            exit
                        end if
                    end do
                    nn = size(o(n_index)%onedarray) - 2          
                    do layer = 1, nn + 1
                        do m = 1, size(&
                        ohat(n_index)%onedarray(layer)%onedarray)
                            do n = 1, size(&
                            delta(n_index)%onedarray(layer)%onedarray)
                            unraveled_der_variables(n_symbol)%&
                            weights(layer)%twodarray(m, n) = &
                            unraveled_der_variables(n_symbol)%&
                            weights(layer)%twodarray(m, n) &
                            + force_coefficient * (2.0d0 / 3.0d0) * &
                            unraveled_variables(&
                            n_symbol)%scaling_slope*&
                            (- nn_atomic_forces(self_index, i)  + &
                            unraveled_real_forces(image_no)%&
                            atomic_forces(self_index, i)) * &
                            der_coordinates_weights_atomic_output(l, i)&
                            %onedarray(layer)%twodarray(m, n) &
                            / no_of_atoms
                            end do
                        end do
                    end do
                    unraveled_der_variables(n_symbol)%scaling_slope = &
                    unraveled_der_variables(n_symbol)%scaling_slope + &
                    force_coefficient * (2.0d0 / 3.0d0) * &
                    der_coordinates_o(l, i)%onedarray(nn + 2)%&
                    onedarray(1) * (- nn_atomic_forces(self_index, i) +&
                    unraveled_real_forces(image_no)%&
                    atomic_forces(self_index, i)) / no_of_atoms
                    do j = 1, size(der_coordinates_o(l, i)%onedarray)
                        deallocate(&
                        der_coordinates_o(l, i)%onedarray(j)%onedarray)
                    end do
                    deallocate(der_coordinates_o(l, i)%onedarray)
                    do j = 1, size(&
                    der_coordinates_weights_atomic_output(l, i)%&
                    onedarray)
                        deallocate(&
                        der_coordinates_weights_atomic_output(l, i)%&
                        onedarray(j)%twodarray)
                    end do
                    deallocate(&
                    der_coordinates_weights_atomic_output(l, i)%&
                    onedarray)
                end do
            end do
            deallocate(n_symbols)
            deallocate(der_coordinates_o)
            deallocate(der_coordinates_weights_atomic_output)
            deallocate(n_self_indices)
        end do
      end if

      do i = 1, size(delta)
        do j = 1, size(delta(i)%onedarray)
            deallocate(delta(i)%onedarray(j)%onedarray)
        end do
        deallocate(delta(i)%onedarray)
      end do

      do i = 1, size(D)
        do j = 1, size(D(i)%onedarray)
            deallocate(D(i)%onedarray(j)%onedarray)
        end do
        deallocate(D(i)%onedarray)
      end do
      end subroutine calculate_der_variables
 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      function ravel_variables(unraveled_der_variables)
 
      type(element_variables_type) :: &
      unraveled_der_variables(no_of_elements)
      double precision :: ravel_variables(len_of_variables)
 
      k = 0
      l = 0
      do symbol = 1, no_of_elements
        if (symbol > 1) then
            k = k + no_layers_of_elements(symbol - 1)
        end if
        do j = 1, no_layers_of_elements(symbol) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            do m = 1, no_of_rows
                do n = 1, no_of_cols
                    ravel_variables(l + (m - 1) * no_of_cols + n) = &
                    unraveled_der_variables(symbol)%weights(j)%&
                    twodarray(m, n)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do symbol = 1, no_of_elements
        ravel_variables(l + 2 *  symbol - 1) = &
        unraveled_der_variables(symbol)%scaling_intercept
        ravel_variables(l + 2 * symbol) = &
        unraveled_der_variables(symbol)%scaling_slope
      end do
      end function ravel_variables
      
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine scale_fingerprints(no_proc_images) 
      
      integer :: no_proc_images
      double precision :: double_temp

      do image_no = 1, no_proc_images
        do index = 1, size(unraveled_fingerprints_of_images(&
        image_no)%onedarray)
            do symbol = 1, no_of_elements
                if (unraveled_atomic_numbers(image_no)%onedarray(index)&
                 == elements_numbers(symbol)) then
                    exit
                end if
            end do    
            do l = 1, len_fingerprints_of_elements(symbol)
                if ((max_fingerprints(symbol, l) - &
                min_fingerprints(symbol, l)) .GT. (10.0d0 ** (-8.0d0)))&
                 then
                    double_temp = unraveled_fingerprints_of_images(&
                    image_no)%onedarray(index)%onedarray(l)
                    double_temp = -1.0d0 + 2.0d0 * (double_temp - &
                    min_fingerprints(symbol, l)) / &
                    (max_fingerprints(symbol, l) - &
                    min_fingerprints(symbol, l))
                    unraveled_fingerprints_of_images(image_no)&
                    %onedarray(index)%onedarray(l) = double_temp
                endif
            end do
        end do
      end do
      end subroutine scale_fingerprints

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine scale_der_fingerprints(no_proc_images) 
      
      integer :: no_proc_images
      integer :: n_symbol
      double precision :: double_temp
      integer, allocatable :: n_self_indices(:)

      do image_no = 1, no_proc_images
        do self_index = 1, size(unraveled_der_fingerprints_of_images(&
        image_no)%onedarray)
            allocate(n_self_indices(size(unraveled_neighborlists(&
            image_no)%onedarray(self_index)%onedarray)))
            do p = 1, size(unraveled_neighborlists(image_no)%&
            onedarray(self_index)%onedarray)
                n_self_indices(p) = unraveled_neighborlists(image_no)%&
                onedarray(self_index)%onedarray(p)
            end do
            do n_index = 1, size(n_self_indices)
                do n_symbol = 1, no_of_elements
                if (unraveled_atomic_numbers(&
                image_no)%onedarray(n_self_indices(n_index)) == &
                elements_numbers(n_symbol)) then
                    exit
                end if
                end do
                do p = 1, 3
                    do q = 1, len_fingerprints_of_elements(n_symbol)
                        if ((max_fingerprints(n_symbol, q) - &
                        min_fingerprints(n_symbol, q)) .GT. &
                        (10.0d0 ** (-8.0d0))) then
                            double_temp = &
                            unraveled_der_fingerprints_of_images(&
                            image_no)%onedarray(self_index)%onedarray(&
                            n_index)%twodarray(p, q)
                            double_temp = 2.0d0 * double_temp / &
                            (max_fingerprints(n_symbol, q) - &
                            min_fingerprints(n_symbol, q))
                            unraveled_der_fingerprints_of_images(&
                            image_no)%onedarray(self_index)%onedarray(&
                            n_index)%twodarray(p, q) = double_temp
                        endif
                    end do
                end do
            end do
            deallocate(n_self_indices)
        end do
      end do

      end subroutine scale_der_fingerprints
 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      end subroutine share_cost_function_task_between_cores
  
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!