!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine calculate_zernike_prime(n, l, n_length, n_indices, &
							numbers, rs, g_numbers, cutoff, indexx, &
							home, p, q, fac_length, factorial, &
							norm_prime)
          implicit none
          integer :: n, l
          integer :: indexx, p, q, n_length, fac_length
          integer, dimension(n_length) :: n_indices, numbers, g_numbers
          double precision, dimension(n_length, 3) :: rs
          double precision, dimension(3) :: home
          double precision, dimension(fac_length) :: factorial
          double precision :: cutoff
          complex*16 :: norm_prime
!f2py     intent(in) :: n, l, n_indices, numbers, g_numbers, rs
!f2py     intent(in) :: home, indexx, p, q, cutoff, n_length, fac_length
!f2py     intent(out) :: norm_prime
          integer :: m
          complex*16 :: c_nlm, c_nlm_prime, z_nlm_, z_nlm, &
          z_nlm_prime, z_nlm_prime_
          integer :: n_index, n_symbol, iter
          double precision, dimension(3) :: neighbor
          double precision :: x, y, z, rho

          norm_prime = (0.0d0, 0.0d0)
          do m = 0, l
			c_nlm = (0.0d0, 0.0d0)
			c_nlm_prime = (0.0d0, 0.0d0)
            do iter = 1, n_length
			  n_index = n_indices(iter)
			  n_symbol = numbers(iter)
			  neighbor(1) = rs(iter, 1)
			  neighbor(2) = rs(iter, 2)
			  neighbor(3) = rs(iter, 3)
			  x = (neighbor(1) - home(1)) / cutoff
			  y = (neighbor(2) - home(2)) / cutoff
			  z = (neighbor(3) - home(3)) / cutoff
			  rho = (x ** 2.0d0 + y ** 2.0d0 + z ** 2.0d0) ** 0.5d0
				  
			  call calculate_z(n, l, m, x, y, z, factorial, &
			  fac_length, z_nlm_)
				  
			  ! Calculate z_nlm
			  z_nlm = z_nlm_ * cutoff_fxn(rho * cutoff, cutoff)
				  
			  ! Calculates z_nlm_prime
			  z_nlm_prime = z_nlm_ * &
			  cutoff_fxn_prime(rho * cutoff, cutoff) * &
			  der_position(indexx, n_index, home, neighbor, p, q)
			  call calculate_z_prime(n, l, m, x, y, z, q, factorial, &
			  fac_length, z_nlm_prime_)
			  if (kronecker(n_index, p) - &
			  kronecker(indexx, p) == 1) then
				z_nlm_prime = z_nlm_prime + &
				cutoff_fxn(rho * cutoff, cutoff) * z_nlm_prime_ / cutoff
			  else if (kronecker(n_index, p) - &
			  kronecker(indexx, p) == -1) then
				z_nlm_prime = z_nlm_prime - &
				cutoff_fxn(rho * cutoff, cutoff) * z_nlm_prime_ / cutoff
			  end if
				  
			  ! sum over neighbors
			  c_nlm = c_nlm + g_numbers(iter) * conjg(z_nlm)
			  c_nlm_prime = c_nlm_prime + &
			  g_numbers(iter) * conjg(z_nlm_prime)
			end do
			! sum over m values
			if (m == 0) then
			  norm_prime = norm_prime + &
			  2.0d0 * c_nlm * conjg(c_nlm_prime)
			else
			  norm_prime = norm_prime + &
			  4.0d0 * c_nlm * conjg(c_nlm_prime)
			end if
		enddo

		CONTAINS

		function cutoff_fxn(r, cutoff)
		  implicit none
		  double precision :: r, cutoff, cutoff_fxn, pi
          if (r > cutoff) then
			cutoff_fxn = 0.0d0
          else
			pi = 4.0d0 * datan(1.0d0)
            cutoff_fxn = 0.5d0 * (cos(pi * r / cutoff) + 1.0d0)
          end if
		end function
      
		function cutoff_fxn_prime(r, cutoff)
		  implicit none
		  double precision :: r, cutoff, cutoff_fxn_prime, pi
		  if (r > cutoff) then
			cutoff_fxn_prime = 0.0d0
          else
			pi = 4.0d0 * datan(1.0d0)
            cutoff_fxn_prime = -0.5d0 * pi * sin(pi*r/cutoff) / cutoff
          end if
		end function

		function der_position(mm, nn, Rm, Rn, ll, ii)
		  implicit none
		  integer :: mm, nn, ll, ii, xyz
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

		function kronecker(i, j)
		  implicit none
		  integer :: i, j
		  integer :: kronecker
		  if (i == j) then
			kronecker = 1
		  else
			kronecker = 0
		  end if
		end function
      
      end subroutine calculate_zernike_prime

	  subroutine calculate_z(n, l, m, x, y, z, factorial, length, &
	  output)
		  implicit none
		  integer :: n, l, m, length
		  double precision :: x, y, z
          double precision, dimension(length) :: factorial
		  complex*16 :: output, ii, term4, term6
!f2py     intent(in) :: n, l, m, x, y, z, factorial, length
!f2py     intent(out) :: output
		  integer :: k, nu, alpha, beta, eta, u, mu, r, s, t
		  double precision :: term1, term2, q, b1, b2, term3
		  double precision :: term5, b5, b6, b7, b8, pi

		  pi = 4.0d0 * datan(1.0d0)

		  output = (0.0d0, 0.0d0)
		  term1 = sqrt((2.0d0 * l + 1.0d0) * &
		  factorial(int(2 * (l + m)) + 1) * &
          factorial(int(2 * (l - m)) + 1)) / factorial(int(2 * l) + 1)
		  term2 = 2.0d0 ** (-m)
		  
		  ii = (0.0d0, 1.0d0)

		  k = int((n - l) / 2.0d0)
		  do nu = 0, k
			call calculate_q(nu, k, l, factorial, length, q)
			do alpha = 0, nu
			  call binomial(float(nu), float(alpha), &
			  factorial, length, b1)
			  do beta = 0, nu - alpha
				call binomial(float(nu - alpha), float(beta), &
				factorial, length, b2)
				term3 = q * b1 * b2
				do u = 0, m
				  call binomial(float(m), float(u), factorial, &
				  length, b5)
				  term4 = ((-1.0d0)**(m - u)) * b5 * (ii**u)
				  do mu = 0, int((l - m) / 2.0d0)
					call binomial(float(l), float(mu), factorial, &
					length, b6)
					call binomial(float(l - mu), float(m + mu), &
					factorial, length, b7)
					term5 = ((-1.0d0) ** mu) * &
					(2.0d0 ** (-2.0d0 * mu)) * b6 * b7
					do eta = 0, mu
					  call binomial(float(mu), float(eta), &
					  factorial, length, b8)
					  r = 2 * (eta + alpha) + u
				      s = 2 * (mu - eta + beta) + m - u
					  t = 2 * (nu - alpha - beta - mu) + l - m
					  output = output + term3 * term4 * term5 * b8 * &
										(x ** r) * (y ** s) * (z ** t)
					end do
				  end do
				end do
			  end do
			end do
		  end do
		  term6 = (ii) ** m
		  output = term1 * term2 * term6 * output
		  output = output / sqrt(4.0d0 * pi / 3.0d0)
	  end subroutine calculate_z

	  subroutine calculate_z_prime(n, l, m, x, y, z, p, factorial, &
									length, output)
		  implicit none
		  integer :: n, l, m, length, p
		  double precision :: x, y, z
          double precision, dimension(length) :: factorial
		  complex*16 :: output, ii, coefficient, term4, term6
!f2py     intent(in) :: n, l, m, x, y, z, factorial, p, length
!f2py     intent(out) :: output
		  integer :: k, nu, alpha, beta, eta, u, mu, r, s, t
		  double precision :: term1, term2, q, b1, b2, term3
		  double precision :: term5, b3, b4, b5, b6, pi

		  pi = 4.0d0 * datan(1.0d0)

		  output = (0.0d0, 0.0d0)
		  term1 = sqrt((2.0d0 * l + 1.0d0) * &
		  factorial(int(2 * (l + m)) + 1) * &
          factorial(int(2 * (l - m)) + 1)) / &
          factorial(int(2 * l) + 1)
		  term2 = 2.0d0 ** (-m)
		  ii = (0.0d0, 1.0d0)

		  k = int((n - l) / 2.)
		  do nu = 0, k
			call calculate_q(nu, k, l, factorial, length, q)
			do alpha = 0, nu
			  call binomial(float(nu), float(alpha), factorial, &
			  length, b1)
			  do beta = 0, nu - alpha
				call binomial(float(nu - alpha), float(beta), &
				factorial, length, b2)
                term3 = q * b1 * b2
                do u = 0, m
                  call binomial(float(m), float(u), factorial, length, &
                  b3) 
                  term4 = ((-1.0d0)**(m - u)) * b3 * (ii**u)
                  do mu = 0, int((l - m) / 2.)
                    call binomial(float(l), float(mu), factorial, &
                    length, b4)
                    call binomial(float(l - mu), float(m + mu), &
                    factorial, length, b5)
                    term5 = &
                    ((-1.0d0)**mu) * (2.0d0**(-2.0d0 * mu)) * b4 * b5
                    do eta = 0, mu
                      call binomial(float(mu), float(eta), factorial, &
                      length, b6)
                      r = 2 * (eta + alpha) + u
                      s = 2 * (mu - eta + beta) + m - u
                      t = 2 * (nu - alpha - beta - mu) + l - m
                      coefficient = term3 * term4 * term5 * b6
                      if (p == 0) then
                        if (r .NE. 0) then
                          output = output + coefficient * r * &
                                   (x ** (r - 1)) * (y ** s) * (z ** t)
                        end if
                      else if (p == 1) then
                        if (s .NE. 0) then
                          output = output + coefficient * s * &
                                   (x ** r) * (y ** (s - 1)) * (z ** t)
                        end if
                      else if (p == 2) then
                        if (t .NE. 0) then
                          output = output + coefficient * t * &
                                   (x ** r) * (y ** s) * (z ** (t - 1))
                        end if
                      end if
                    end do
                  end do
                end do
              end do
            end do
          end do
		  term6 = (ii) ** m
		  output = term1 * term2 * term6 * output
		  output = output / sqrt(4.0d0 * pi / 3.0d0)
	  end subroutine calculate_z_prime

	  subroutine calculate_q(nu, k, l, factorial, length, output)
		  implicit none
		  integer :: nu, k, l, length
          double precision, dimension(length) :: factorial
		  double precision :: output, b1, b2, b3, b4
!f2py     intent(in) :: nu, k, l, factorial
!f2py     intent(out) :: output

		  call binomial(float(k), float(nu), factorial, length, b1)
		  call binomial(float(2 * k), float(k), factorial, length, b2)
		  call binomial(float(2 * (k + l + nu) + 1), float(2 * k), &
		  factorial, length, b3)
		  call binomial(float(k + l + nu), float(k), factorial, &
		  length, b4)
		  output = ((-1.0d0) ** (k + nu)) * &
		  sqrt((2.0d0 * l + 4.0d0 * k + 3.0d0) / 3.0d0) * b1 * b2 * &
          b3 / b4 / (2.0d0 ** (2.0d0 * k))
	  end subroutine calculate_q

	  subroutine binomial(n, k, factorial, length, output)
		  implicit none
		  real(4) :: n, k
          integer :: length
          double precision, dimension(length) :: factorial
		  double precision :: output
!f2py     intent(in) :: n, k, factorial, length
!f2py     intent(out) :: output
        output = factorial(INT(2 * n) + 1) / &
        factorial(INT(2 * k) + 1) / &
        factorial(INT(2 * (n - k)) + 1)
	  end subroutine binomial

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
