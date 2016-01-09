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
                           g_eta, cutoff, home, tag, n, ridge)
               
              implicit none
              integer, dimension(n) :: numbers
              integer, dimension(2) :: g_numbers
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home
              integer :: n, tag
              double precision :: g_gamma, g_zeta, g_eta, cutoff
              double precision :: ridge
!f2py         intent(in) :: numbers, rs, g_numbers, g_gamma, g_zeta,tag
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
                    if (tag == 1) then
                        term = term*cutoff_fxn(Rij, cutoff)
                        term = term*cutoff_fxn(Rik, cutoff)
                        term = term*cutoff_fxn(Rjk, cutoff)
                    elseif (tag == 2) then
                        term = term*(1.0d0/3.0d0)*&
                        (cutoff_fxn(Rij, cutoff)*&
                        cutoff_fxn(Rik, cutoff)+&
                        cutoff_fxn(Rij, cutoff)*&
                        cutoff_fxn(Rjk, cutoff)+&
                        cutoff_fxn(Rik, cutoff)*&
                        cutoff_fxn(Rjk, cutoff))
                    endif 
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
                            g_gamma, g_zeta, g_eta, &
                            cutoff, aa, home, mm, ii, tag, n, ridge)
               
              implicit none
              integer, dimension(n) :: n_indices, numbers
              integer, dimension(2) :: g_numbers
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home, Rj, Rk
              integer :: n, aa, mm, ii, tag
              double precision :: g_gamma, g_zeta, g_eta, cutoff
              double precision :: ridge
!f2py         intent(in) :: numbers, rs, g_numbers, g_gamma, g_zeta
!f2py         intent(in) :: g_eta, cutoff, home, n_indices , aa, mm, ii, tag
!f2py         intent(hide) :: n
!f2py         intent(out) :: ridge
              integer :: j, k, match, xyz
              double precision, dimension(3) :: Raj_, Rak_, Rjk_
              double precision :: Raj, Rak, Rjk, costheta
              double precision :: c1, c2, c3, c4
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
                    if (tag == 1) then
                        term2 = c2 * c3 * c4
                    elseif (tag == 2) then
                        term2 = (1.0d0/3.0d0) * &
                        (c2 * c3 + c2 * c4 + c3 * c4)
                    endif 
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
                    if (tag == 1) then
                        term8 = &
                        der_cutoff_fxn(Raj, cutoff) * c3 * c4 * term5
                        term9 = &
                        c2 * der_cutoff_fxn(Rak, cutoff) * c4 * term6
                        term10 = &
                        c2 * c3 * der_cutoff_fxn(Rjk, cutoff) * term7
                    elseif (tag == 2) then
                        term8 = der_cutoff_fxn(Raj, cutoff) * term5 * c3
                        term8 = &
                        term8 + c2 * der_cutoff_fxn(Rak, cutoff) * term6
                        term8 = (1.0d0/3.0d0) * term8
                        term9 = der_cutoff_fxn(Raj, cutoff) * term5 * c4
                        term9 = &
                        term9 + c2 * der_cutoff_fxn(Rjk, cutoff) * term7
                        term9 = (1.0d0/3.0d0) * term9
                        term10 = &
                        der_cutoff_fxn(Rak, cutoff) * term6 * c4
                        term10 = term10 + &
                        c3 * der_cutoff_fxn(Rjk, cutoff) * term7
                        term10 = (1.0d0/3.0d0) * term10
                    endif
                    term11 = term2 + c1 * (term8 + term9 + term10)
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