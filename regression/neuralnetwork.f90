!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module that utilizes the regression model to calculate energies and forces as well as their derivatives
!     function names ending with an underscore correspond to no-fingerprinting scheme.

      module regression
      implicit none
      
!    the data of regression (should be feed in by python)
      integer, allocatable:: no_layers_of_elements(:)
      integer, allocatable:: no_nodes_of_elements(:)
      integer :: activation_signal
      
      type:: real_two_d_array
        sequence
        double precision, allocatable:: twodarray(:,:)
      end type real_two_d_array
      
      type:: element_variables
        sequence
        double precision:: intercept
        double precision:: slope
        type(real_two_d_array), allocatable:: weights(:)
      end type element_variables

      type:: real_one_d_array
        sequence
        double precision, allocatable:: onedarray(:)
      end type real_one_d_array
      
      contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns energy value in the no-fingerprinting scheme 
      function get_energy_(len_of_input, input, len_of_variables, &
      variables)
      implicit none
      
      integer :: len_of_input, len_of_variables
      double precision :: input(len_of_input)
      double precision :: variables(len_of_variables)
      double precision :: get_energy_
   
      integer:: p, m, n, layer
      integer:: l, j, no_of_rows, no_of_cols, q
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope

!     changing the form of variables from vector into derived-types 
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(no_of_rows, no_of_cols))
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  weights(j)%twodarray(p, q) = &
                  variables(l + (p - 1) * no_of_cols + q)
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do
      intercept = variables(l + 1)
      slope = variables(l + 2)

      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do

      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(len_of_input))
      allocate(ohat(1)%onedarray(len_of_input + 1))
      do m = 1, len_of_input
          o(1)%onedarray(m) = input(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) &
                  * weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = &
                  tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1) =  1.0
          deallocate(net)
      end do
      
      get_energy_ = slope * o(layer)%onedarray(1) + intercept

!     deallocating neural network
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      
!     deallocating derived type variables
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)

      end function get_energy_

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns energy value in the fingerprinting scheme 
      function get_energy(symbol, &
      len_of_fingerprint, fingerprint, &
      no_of_elements, elements_numbers, &
      len_of_variables, variables)
      implicit none
      
      integer :: symbol, len_of_variables, &
      len_of_fingerprint, no_of_elements
      double precision :: fingerprint(len_of_fingerprint)
      integer :: elements_numbers(no_of_elements)
      double precision :: variables(len_of_variables)
      double precision :: get_energy
   
      integer:: p, element, m, n, layer
      integer:: k, l, j, no_of_rows, no_of_cols, q
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(element_variables):: unraveled_variables(no_of_elements)

!     changing the form of variables from vector into derived-types 
      k = 0
      l = 0
      do element = 1, no_of_elements
        allocate(unraveled_variables(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_variables(&
            element)%weights(j)%twodarray(no_of_rows, no_of_cols))
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    unraveled_variables(element)%weights(j)%twodarray(&
                    p, q) = variables(l + (p - 1) * no_of_cols + q)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do element = 1, no_of_elements
        unraveled_variables(element)%intercept = &
        variables(l + 2 *  element - 1)
        unraveled_variables(element)%slope = variables(l + 2 * element)
      end do

      p = 0
      do element = 1, no_of_elements
          if (symbol == elements_numbers(element)) then
              exit
          else 
              p = p + no_layers_of_elements(element)
          end if
      end do
      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do

      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_variables(element)%weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_variables(&
                  element)%weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0
          deallocate(net)
      end do

      get_energy = unraveled_variables(element)%slope * &
      o(layer)%onedarray(1) + unraveled_variables(element)%intercept

!     deallocating neural network
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)

!      deallocating derived type variables
      do element = 1, no_of_elements
        deallocate(unraveled_variables(element)%weights)
      end do

      end function get_energy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns force values in the no-fingerprinting scheme 
      function get_force_(len_of_input, input, input_, &
      len_of_variables, variables)
      implicit none
      
      integer :: len_of_input, len_of_variables
      double precision :: input(len_of_input)
      double precision :: input_(len_of_input)
      double precision :: variables(len_of_variables)
      double precision :: get_force_
   
      double precision, allocatable:: temp(:)
      integer:: p, q, m, n, nn, layer
      integer:: l, j, no_of_rows, no_of_cols
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: der_coordinates_o(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope

!     changing the form of variables to derived-types
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(no_of_rows, no_of_cols))
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  weights(j)%twodarray(p, q) = &
                  variables(l + (p - 1) * no_of_cols + q)
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do
      
      intercept = variables(l + 1)
      slope = variables(l + 2)

      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do
      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(len_of_input))
      allocate(ohat(1)%onedarray(len_of_input + 1))
      do m = 1, len_of_input
          o(1)%onedarray(m) = input(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * &
                  weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          deallocate(net)
      end do
      nn = size(o) - 2
      allocate(der_coordinates_o(nn + 2))
      allocate(der_coordinates_o(1)%onedarray(len_of_input))
      do m = 1, len_of_input
      der_coordinates_o(1)%onedarray(m) = input_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(weights(layer)%twodarray, dim = 2)))
          do p = 1, size(weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + der_coordinates_o(&
                  layer)%onedarray(q) * weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(der_coordinates_o(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                der_coordinates_o(layer + 1)%onedarray(p) = &
                temp(p) * (1.0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                der_coordinates_o(layer + 1)%onedarray(p) = &
                temp(p) * (1.0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
                der_coordinates_o(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do

      get_force_ = - slope * der_coordinates_o(nn + 2)%onedarray(1)
!     deallocating neural network
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(der_coordinates_o)
          deallocate(der_coordinates_o(p)%onedarray)
      end do
      deallocate(der_coordinates_o)

!     deallocating derived type variables 
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)
   
      end function get_force_

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns force values in the fingerprinting scheme 
      function get_force(n_symbol, len_of_fingerprint, fingerprint, &
      der_fingerprint, no_of_elements, elements_numbers, &
      len_fingerprints_of_elements, len_of_variables, variables)
      implicit none
      
      integer :: n_symbol, len_of_fingerprint, len_of_variables
      integer :: no_of_elements
      double precision :: fingerprint(len_of_fingerprint)
      double precision :: der_fingerprint(len_of_fingerprint)
      integer :: elements_numbers(no_of_elements)
      integer :: len_fingerprints_of_elements(no_of_elements)
      double precision :: variables(len_of_variables)
      double precision :: get_force
   
      double precision, allocatable:: temp(:)
      integer:: p, q, element, m, n, nn, layer
      integer:: k, l, j, no_of_rows, no_of_cols
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: der_coordinates_o(:)
      type(element_variables):: unraveled_variables(no_of_elements)

!     changing the form of variables to derived-types
      k = 0
      l = 0
      do element = 1, no_of_elements
        allocate(unraveled_variables(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_variables(&
            element)%weights(j)%twodarray(no_of_rows, no_of_cols))
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    unraveled_variables(element)%weights(j)%twodarray(&
                    p, q) = variables(l + (p - 1) * no_of_cols + q)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do element = 1, no_of_elements
        unraveled_variables(element)%intercept = &
        variables(l + 2 *  element - 1)
        unraveled_variables(element)%slope = variables(l + 2 * element)
      end do
 
      p = 0
      do element = 1, no_of_elements
          if (n_symbol == elements_numbers(element)) then
              exit
          else 
              p = p + no_layers_of_elements(element)
          end if
      end do

      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_variables(element)%weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_variables(&
                  element)%weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          deallocate(net)
      end do
      nn = size(o) - 2
      allocate(der_coordinates_o(nn + 2))
      allocate(der_coordinates_o(1)%onedarray(&
      len_fingerprints_of_elements(element)))
      do m = 1, len_fingerprints_of_elements(element)
      der_coordinates_o(1)%onedarray(m) = der_fingerprint(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim = 2)))
          do p = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(unraveled_variables(element)%weights(&
              layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + der_coordinates_o(&
                  layer)%onedarray(q) * unraveled_variables(&
                  element)%weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(der_coordinates_o(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                der_coordinates_o(layer + 1)%onedarray(p) = temp(p) * &
                (1.0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                der_coordinates_o(layer + 1)%onedarray(p) = &
                temp(p) * (1.0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
                der_coordinates_o(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do

      get_force = - unraveled_variables(element)%slope * &
      der_coordinates_o(nn + 2)%onedarray(1)
!     deallocating neural network
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(der_coordinates_o)
          deallocate(der_coordinates_o(p)%onedarray)
      end do
      deallocate(der_coordinates_o)

!     deallocating derived type variables 
      do element = 1, no_of_elements
        deallocate(unraveled_variables(element)%weights)
      end do
   
      end function get_force

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns variable derivative of the energy cost function in the no-fingerprinting scheme       
      function get_variable_der_of_energy_(len_of_input, input, &
      len_of_variables, variables)
      implicit none
      
      integer:: len_of_input, len_of_variables    
      double precision:: get_variable_der_of_energy_(len_of_variables)
      double precision :: variables(len_of_variables)
      double precision :: input(len_of_input)
      
      integer:: m, n, j, l, layer, p, q, nn, no_of_cols, no_of_rows
      double precision:: temp1, temp2
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope
      type(real_two_d_array), allocatable:: der_weights(:)
      double precision:: der_intercept
      double precision:: der_slope

!     changing the form of variables from vector into derived-types
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(no_of_rows, no_of_cols))
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  weights(j)%twodarray(p, q) = &
                  variables(l + (p - 1) * no_of_cols + q)
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do
      intercept = variables(l + 1)
      slope = variables(l + 2)

      der_intercept = 0.d0
      der_slope = 0.d0
      l = 0
      allocate(der_weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          allocate(der_weights(j)%twodarray(no_of_rows, no_of_cols))
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  der_weights(j)%twodarray(p, q) = 0.0d0
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do

      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do
      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(len_of_input))
      allocate(ohat(1)%onedarray(len_of_input + 1))
      do m = 1, len_of_input
          o(1)%onedarray(m) = input(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * weights(&
                  layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1) =  1.0
          deallocate(net)
      end do

      nn = size(o) - 2
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = &
                  (1.0d0 - o(layer + 1)%onedarray(j)* &
                  o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
                  D(layer)%onedarray(j) = 1.0d0
              end if
          end do    
      end do
      allocate(delta(nn + 1))
      allocate(delta(nn + 1)%onedarray(1))
      delta(nn + 1)%onedarray(1) = D(nn + 1)%onedarray(1)
      do layer = nn, 1, -1
          allocate(delta(layer)%onedarray(size(D(layer)%onedarray)))
          do p = 1, size(D(layer)%onedarray)
              delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp1 = D(layer)%onedarray(p) * &
                  weights(layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      der_intercept = 1.0d0
      der_slope = o(nn + 2)%onedarray(1)
      do layer = 1, nn + 1
          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
                  der_weights(layer)%twodarray(p, q) = slope * &
                  ohat(layer)%onedarray(p) * delta(layer)%onedarray(q)
              end do
          end do
      end do

      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(delta)
          deallocate(delta(p)%onedarray)
      end do
      deallocate(delta)
      do p = 1, size(D)
          deallocate(D(p)%onedarray)
      end do
      deallocate(D)

!      changing the derivatives of the cost function from derived-type form into vector
      l = 0
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  get_variable_der_of_energy_(&
                  l + (p - 1) * no_of_cols + q) = &
                  der_weights(j)%twodarray(p, q)
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do

      get_variable_der_of_energy_(l + 1) = der_intercept
      get_variable_der_of_energy_(l + 2) = der_slope

!     deallocating derived-type variables 
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)
      do p = 1, size(der_weights)
          deallocate(der_weights(p)%twodarray)
      end do
      deallocate(der_weights)
 
      end function get_variable_der_of_energy_
   
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns variable derivative of the energy cost function in the fingerprinting scheme         
      function get_variable_der_of_energy(symbol, len_of_fingerprint, &
      fingerprint, no_of_elements, elements_numbers, &
      len_of_variables, variables)
      implicit none
      
      integer:: len_of_variables, no_of_elements   
      integer:: symbol, len_of_fingerprint
      double precision:: get_variable_der_of_energy(len_of_variables)
      double precision :: variables(len_of_variables)
      double precision :: fingerprint(len_of_fingerprint)
      integer :: elements_numbers(no_of_elements)
      
      integer:: element, m, n, j, k, l, layer, p, q, nn, no_of_cols
      integer:: no_of_rows
      double precision:: temp1, temp2
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      type(element_variables):: unraveled_variables(no_of_elements)
      type(element_variables):: unraveled_der_variables(no_of_elements)

!     changing the form of variables to derived types
      k = 0
      l = 0
      do element = 1, no_of_elements
        allocate(unraveled_variables(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_variables(element)%weights(j)%twodarray(&
            no_of_rows, no_of_cols))
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    unraveled_variables(element)%weights(j)%twodarray(&
                    p, q) = variables(l + (p - 1) * no_of_cols + q)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do element = 1, no_of_elements
        unraveled_variables(element)%intercept = &
        variables(l + 2 *  element - 1)
        unraveled_variables(element)%slope = variables(l + 2 * element)
      end do

      do element = 1, no_of_elements
        unraveled_der_variables(element)%intercept = 0.d0
        unraveled_der_variables(element)%slope = 0.d0
      end do
      k = 0
      l = 0
      do element = 1, no_of_elements
        allocate(unraveled_der_variables(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_der_variables(&
            element)%weights(j)%twodarray(no_of_rows, no_of_cols))
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    unraveled_der_variables(&
                    element)%weights(j)%twodarray(p, q) = 0.0d0
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
 
      p = 0
      do element = 1, no_of_elements
          if (symbol == elements_numbers(element)) then
              exit
          else 
              p = p + no_layers_of_elements(element)
          end if
      end do
      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_variables(element)%weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_variables(&
                  element)%weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0
          deallocate(net)
      end do
 
      nn = size(o) - 2
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = (1.0d0 - &
                  o(layer + 1)%onedarray(j)* o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
                  D(layer)%onedarray(j) = 1.0d0
              end if
          end do    
      end do
      allocate(delta(nn + 1))
      allocate(delta(nn + 1)%onedarray(1))
      delta(nn + 1)%onedarray(1) = D(nn + 1)%onedarray(1)
      do layer = nn, 1, -1
          allocate(delta(layer)%onedarray(size(D(layer)%onedarray)))
          do p = 1, size(D(layer)%onedarray)
              delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp1 = D(layer)%onedarray(p) * unraveled_variables(&
                  element)%weights(layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do
    
      unraveled_der_variables(element)%intercept = 1.0d0
      unraveled_der_variables(element)%slope = o(nn + 2)%onedarray(1)
      do layer = 1, nn + 1
          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
                  unraveled_der_variables(element)%weights(&
                  layer)%twodarray(p, q) = &
                  unraveled_variables(element)%slope * &
                  ohat(layer)%onedarray(p) * delta(layer)%onedarray(q) 
              end do
          end do
      end do
  
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(delta)
          deallocate(delta(p)%onedarray)
      end do
      deallocate(delta)
      do p = 1, size(D)
          deallocate(D(p)%onedarray)
      end do
      deallocate(D)

!      changing the derivatives of the cost function from derived-type form into vector
      k = 0
      l = 0
      do element = 1, no_of_elements
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    get_variable_der_of_energy(&
                    l + (p - 1) * no_of_cols + q) = &
                    unraveled_der_variables(&
                    element)%weights(j)%twodarray(p, q)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do element = 1, no_of_elements
        get_variable_der_of_energy(l + 2 *  element - 1) = &
        unraveled_der_variables(element)%intercept
        get_variable_der_of_energy(l + 2 * element) = &
        unraveled_der_variables(element)%slope
      end do

!     deallocating derived-type variables 
      do element = 1, no_of_elements
        deallocate(unraveled_variables(element)%weights)
        deallocate(unraveled_der_variables(element)%weights)
      end do
 
      end function get_variable_der_of_energy

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns variable derivative of the force cost function in the no-fingerprinting scheme   
      function get_variable_der_of_forces_(len_of_input, input, &
      input_, len_of_variables, variables)
      implicit none
      
      integer:: len_of_input, len_of_variables    
      double precision:: get_variable_der_of_forces_(len_of_variables)
      double precision :: variables(len_of_variables)
      double precision :: input(len_of_input)
      double precision :: input_(len_of_input)

      integer:: m, n, j, l, layer, p, q, nn, no_of_cols
      integer:: no_of_rows
      double precision:: temp1, temp2
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      type(real_one_d_array), allocatable:: der_coordinates_o(:)
      double precision, allocatable:: der_coordinates_ohat(:)
      type(real_one_d_array), allocatable:: der_coordinates_D(:)
      type (real_one_d_array), allocatable:: der_coordinates_delta(:)
      double precision, allocatable:: &
      der_coordinates_weights_atomic_output(:, :)
      double precision, allocatable:: temp(:), temp3(:), temp4(:)
      double precision, allocatable:: temp5(:), temp6(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope
      type(real_two_d_array), allocatable:: der_weights(:)
      double precision:: der_intercept
      double precision:: der_slope

!     changing the form of variables from vector into derived-types 
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(no_of_rows, no_of_cols))
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  weights(j)%twodarray(p, q) = &
                  variables(l + (p - 1) * no_of_cols + q)
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do
      intercept = variables(l + 1)
      slope = variables(l + 2)
      
      der_intercept = 0.d0
      der_slope = 0.d0
      l = 0
      allocate(der_weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          allocate(der_weights(j)%twodarray(no_of_rows, no_of_cols))
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  der_weights(j)%twodarray(p, q) = 0.0d0
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do
      
      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do
      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(len_of_input))
      allocate(ohat(1)%onedarray(len_of_input + 1))
      do m = 1, len_of_input
          o(1)%onedarray(m) = input(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * &
                  weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1) =  1.0
          deallocate(net)
      end do

      nn = size(o) - 2
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = (1.0d0 - &
                  o(layer + 1)%onedarray(j)* o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
                  D(layer)%onedarray(j) = 1.0d0
              end if
          end do    
      end do
      allocate(delta(nn + 1))
      allocate(delta(nn + 1)%onedarray(1))
      delta(nn + 1)%onedarray(1) = D(nn + 1)%onedarray(1)
      do layer = nn, 1, -1
          allocate(delta(layer)%onedarray(size(D(layer)%onedarray)))
          do p = 1, size(D(layer)%onedarray)
              delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp1 = D(layer)%onedarray(p) * weights(&
                  layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      allocate(der_coordinates_o(nn + 2))
      allocate(der_coordinates_o(1)%onedarray(len_of_input))
      do m = 1, len_of_input
        der_coordinates_o(1)%onedarray(m) = input_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(weights(layer)%twodarray, dim = 2)))
          do p = 1, size(weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + der_coordinates_o(&
                  layer)%onedarray(q) * weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(der_coordinates_o(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                der_coordinates_o(layer + 1)%onedarray(p) = temp(p) * &
                (1.0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                der_coordinates_o(layer + 1)%onedarray(p) = &
                temp(p) * (1.0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
                der_coordinates_o(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do  

      allocate(der_coordinates_D(nn + 1))
      do layer = 1, nn + 1
          allocate(der_coordinates_D(layer)%onedarray(&
          size(o(layer + 1)%onedarray)))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  der_coordinates_D(layer)%onedarray(p) = &
                  - 2.0d0 * o(layer + 1)%onedarray(p) * &
                  der_coordinates_o(layer + 1)%onedarray(p)
              elseif (activation_signal == 2) then
                  der_coordinates_D(layer)%onedarray(p) = &
                  der_coordinates_o(layer + 1)%onedarray(p) * &
                  (1.0d0 - 2.0d0 * o(layer + 1)%onedarray(p))
              elseif (activation_signal == 3) then
                  der_coordinates_D(layer)%onedarray(p) =0.0d0
              end if
          end do    
      end do

      allocate(der_coordinates_delta(nn + 1))
      allocate(der_coordinates_delta(nn + 1)%onedarray(1))
      der_coordinates_delta(nn + 1)%onedarray(1) = &
      der_coordinates_D(nn + 1)%onedarray(1)

      do layer = nn, 1, -1
          allocate(temp3(&
          size(weights(layer + 1)%twodarray, dim = 1) - 1))
          allocate(temp4(&
          size(weights(layer + 1)%twodarray, dim = 1) - 1))
          do p = 1, size(weights(layer + 1)%twodarray, dim = 1) - 1
              temp3(p) = 0.0d0
              temp4(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp3(p) = temp3(p) + weights(layer + 1)%twodarray(&
                  p, q) * delta(layer + 1)%onedarray(q)
                  temp4(p) = temp4(p) + weights(layer + 1)%twodarray(&
                  p, q) * der_coordinates_delta(layer + 1)%onedarray(q)
              end do
          end do
          allocate(temp5(size(der_coordinates_D(layer)%onedarray)))
          allocate(temp6(size(der_coordinates_D(layer)%onedarray)))
          allocate(der_coordinates_delta(layer)%onedarray(&
          size(der_coordinates_D(layer)%onedarray)))
          do p = 1, size(der_coordinates_D(layer)%onedarray)
              temp5(p) = &
              der_coordinates_D(layer)%onedarray(p) * temp3(p)
              temp6(p) = D(layer)%onedarray(p) * temp4(p)
              der_coordinates_delta(layer)%onedarray(p)= &
              temp5(p) + temp6(p)
          end do
          deallocate(temp3)
          deallocate(temp4)
          deallocate(temp5)
          deallocate(temp6)
      end do
      
      der_slope = der_coordinates_o(nn + 2)%onedarray(1)
      do layer = 1, nn + 1
          allocate(der_coordinates_ohat(&
          size(der_coordinates_o(layer)%onedarray) + 1))
          do p = 1, size(der_coordinates_o(layer)%onedarray)
              der_coordinates_ohat(p) = &
              der_coordinates_o(layer)%onedarray(p)
          end do
          der_coordinates_ohat(&
          size(der_coordinates_o(layer)%onedarray) + 1) = 0.0d0
          allocate(der_coordinates_weights_atomic_output(&
          size(der_coordinates_ohat), size(delta(layer)%onedarray)))
          do p = 1, size(der_coordinates_ohat)
              do q = 1, size(delta(layer)%onedarray)
                  der_coordinates_weights_atomic_output(p, q)= 0.0d0
              end do
          end do
          do p = 1, size(der_coordinates_ohat)
              do q = 1, size(delta(layer)%onedarray)
              der_coordinates_weights_atomic_output(p, q) = &
              der_coordinates_weights_atomic_output(p, q) + &
              der_coordinates_ohat(p) * delta(layer)%onedarray(q) + &
              ohat(layer)%onedarray(p)* &
              der_coordinates_delta(layer)%onedarray(q)
              end do
          end do 
          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
              der_weights(layer)%twodarray(p, q) = &
              slope * der_coordinates_weights_atomic_output(p, q)
              end do
          end do
          deallocate(der_coordinates_ohat)
          deallocate(der_coordinates_weights_atomic_output)
      end do

!     deallocating neural network 
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(delta)
          deallocate(delta(p)%onedarray)
      end do
      deallocate(delta)
      do p = 1, size(D)
          deallocate(D(p)%onedarray)
      end do
      deallocate(D)
      do p = 1, size(der_coordinates_o)
          deallocate(der_coordinates_o(p)%onedarray)
      end do
      deallocate(der_coordinates_o)
      do p = 1, size(der_coordinates_delta)
          deallocate(der_coordinates_delta(p)%onedarray)
      end do
      deallocate(der_coordinates_delta)
      do p = 1, size(der_coordinates_D)
          deallocate(der_coordinates_D(p)%onedarray)
      end do
      deallocate(der_coordinates_D)

      l = 0
      do j = 1, no_layers_of_elements(1) - 1
          no_of_rows = no_nodes_of_elements(j) + 1
          no_of_cols = no_nodes_of_elements(j + 1)
          do p = 1, no_of_rows
              do q = 1, no_of_cols
                  get_variable_der_of_forces_(&
                  l + (p - 1) * no_of_cols + q) = &
                  der_weights(j)%twodarray(p, q)
              end do
          end do
          l = l + no_of_rows * no_of_cols
      end do
      get_variable_der_of_forces_(l + 1) = der_intercept
      get_variable_der_of_forces_(l + 2) = der_slope

!     deallocating derived-type variables      
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)
      do p = 1, size(der_weights)
          deallocate(der_weights(p)%twodarray)
      end do
      deallocate(der_weights)
      
      end function get_variable_der_of_forces_

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns variable derivative of the force cost function in the fingerprinting scheme   
      function get_variable_der_of_forces(n_symbol, &
      len_of_fingerprint, fingerprint, der_fingerprint, &
      no_of_elements, elements_numbers, &
      len_fingerprints_of_elements, &
      len_of_variables, variables)
      implicit none
      
      integer :: n_symbol, len_of_fingerprint
      integer :: len_of_variables, no_of_elements
      double precision :: fingerprint(len_of_fingerprint)
      double precision :: der_fingerprint(len_of_fingerprint)
      integer :: len_fingerprints_of_elements(no_of_elements)
      integer :: elements_numbers(no_of_elements)
      double precision :: variables(len_of_variables)
      double precision:: get_variable_der_of_forces(len_of_variables)

      integer:: element, m, n, j, k, l, layer, p, q, nn, no_of_cols
      integer:: no_of_rows
      double precision:: temp1, temp2
      integer, allocatable :: hiddensizes(:)
      double precision, allocatable :: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      type(real_one_d_array), allocatable:: der_coordinates_o(:)
      double precision, allocatable:: der_coordinates_ohat(:)
      type(real_one_d_array), allocatable:: der_coordinates_D(:)
      type (real_one_d_array), allocatable:: der_coordinates_delta(:)
      double precision, allocatable:: &
      der_coordinates_weights_atomic_output(:, :)
      double precision, allocatable:: temp(:), temp3(:), temp4(:)
      double precision, allocatable:: temp5(:), temp6(:)
      type(element_variables):: unraveled_variables(no_of_elements)
      type(element_variables):: unraveled_der_variables(no_of_elements)

!     changing the form of variables from vector into derived-types 
      k = 0
      l = 0
      do element = 1, no_of_elements
        allocate(unraveled_variables(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_variables(&
            element)%weights(j)%twodarray(no_of_rows, no_of_cols))
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    unraveled_variables(element)%weights(j)%twodarray(&
                    p, q) = variables(l + (p - 1) * no_of_cols + q)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do element = 1, no_of_elements
        unraveled_variables(element)%intercept = &
        variables(l + 2 *  element - 1)
        unraveled_variables(element)%slope = variables(l + 2 * element)
      end do
      
      do element = 1, no_of_elements
        unraveled_der_variables(element)%intercept = 0.d0
        unraveled_der_variables(element)%slope = 0.d0
      end do
      k = 0
      l = 0
      do element = 1, no_of_elements
        allocate(unraveled_der_variables(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_der_variables(&
            element)%weights(j)%twodarray(no_of_rows, no_of_cols))
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    unraveled_der_variables(&
                    element)%weights(j)%twodarray(p, q) = 0.0d0
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do

      p = 0
      do element = 1, no_of_elements
          if (n_symbol == elements_numbers(element)) then
              exit
          else 
              p = p + no_layers_of_elements(element)
          end if
      end do
      
      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_variables(element)%weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_variables(element)%weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * &
                  unraveled_variables(element)%weights(&
                  layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1. / (1. +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0
          deallocate(net)
      end do

      nn = size(o) - 2
      
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = &
                  (1.0d0 - o(layer + 1)%onedarray(j)* &
                  o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
                  D(layer)%onedarray(j) = 1.0d0
              end if
          end do    
      end do
      allocate(delta(nn + 1))
      allocate(delta(nn + 1)%onedarray(1))
      delta(nn + 1)%onedarray(1) = D(nn + 1)%onedarray(1)
      do layer = nn, 1, -1
          allocate(delta(layer)%onedarray(size(D(layer)%onedarray)))
          do p = 1, size(D(layer)%onedarray)
              delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp1 = D(layer)%onedarray(p) * &
                  unraveled_variables(element)%weights(&
                  layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      allocate(der_coordinates_o(nn + 2))
      allocate(der_coordinates_o(1)%onedarray(&
      len_fingerprints_of_elements(element)))
      do m = 1, len_fingerprints_of_elements(element)
        der_coordinates_o(1)%onedarray(m) = der_fingerprint(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim = 2)))
          do p = 1, size(unraveled_variables(&
          element)%weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(unraveled_variables(&
              element)%weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + der_coordinates_o(&
                  layer)%onedarray(q) * unraveled_variables(&
                  element)%weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(der_coordinates_o(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                der_coordinates_o(layer + 1)%onedarray(p) = temp(p) * &
                (1.0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                der_coordinates_o(layer + 1)%onedarray(p) = temp(p) * &
                (1.0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
                der_coordinates_o(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do  

      allocate(der_coordinates_D(nn + 1))
      do layer = 1, nn + 1
          allocate(der_coordinates_D(layer)%onedarray(&
          size(o(layer + 1)%onedarray)))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  der_coordinates_D(layer)%onedarray(p) =- 2.0d0 * &
                  o(layer + 1)%onedarray(p) * &
                  der_coordinates_o(layer + 1)%onedarray(p)
              elseif (activation_signal == 2) then
                  der_coordinates_D(layer)%onedarray(p) = &
                  der_coordinates_o(layer + 1)%onedarray(p) * &
                  (1.0d0 - 2.0d0 * o(layer + 1)%onedarray(p))
              elseif (activation_signal == 3) then
                  der_coordinates_D(layer)%onedarray(p) =0.0d0
              end if
          end do    
      end do

      allocate(der_coordinates_delta(nn + 1))
      allocate(der_coordinates_delta(nn + 1)%onedarray(1))
      der_coordinates_delta(nn + 1)%onedarray(1) = &
      der_coordinates_D(nn + 1)%onedarray(1)

      do layer = nn, 1, -1
          allocate(temp3(size(unraveled_variables(element)%weights(&
          layer + 1)%twodarray, dim = 1) - 1))
          allocate(temp4(size(unraveled_variables(element)%weights(&
          layer + 1)%twodarray, dim = 1) - 1))
          do p = 1, size(unraveled_variables(element)%weights(&
          layer + 1)%twodarray, dim = 1) - 1
              temp3(p) = 0.0d0
              temp4(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp3(p) = temp3(p) + unraveled_variables(&
                  element)%weights(layer + 1)%twodarray(p, q) * &
                  delta(layer + 1)%onedarray(q)
                  temp4(p) = temp4(p) + unraveled_variables(&
                  element)%weights(layer + 1)%twodarray(p, q) * &
                  der_coordinates_delta(layer + 1)%onedarray(q)
              end do
          end do
          allocate(temp5(size(der_coordinates_D(layer)%onedarray)))
          allocate(temp6(size(der_coordinates_D(layer)%onedarray)))
          allocate(der_coordinates_delta(layer)%onedarray(&
          size(der_coordinates_D(layer)%onedarray)))
          do p = 1, size(der_coordinates_D(layer)%onedarray)
              temp5(p) = &
              der_coordinates_D(layer)%onedarray(p) * temp3(p)
              temp6(p) = D(layer)%onedarray(p) * temp4(p)
              der_coordinates_delta(layer)%onedarray(p)= &
              temp5(p) + temp6(p)
          end do
          deallocate(temp3)
          deallocate(temp4)
          deallocate(temp5)
          deallocate(temp6)
      end do
      
      unraveled_der_variables(element)%slope = &
      der_coordinates_o(nn + 2)%onedarray(1)
      do layer = 1, nn + 1
          allocate(der_coordinates_ohat(&
          size(der_coordinates_o(layer)%onedarray) + 1))
          do p = 1, size(der_coordinates_o(layer)%onedarray)
              der_coordinates_ohat(p) = &
              der_coordinates_o(layer)%onedarray(p)
          end do
          der_coordinates_ohat(&
          size(der_coordinates_o(layer)%onedarray) + 1) = 0.0d0
          allocate(der_coordinates_weights_atomic_output(&
          size(der_coordinates_ohat), size(delta(layer)%onedarray)))
          do p = 1, size(der_coordinates_ohat)
              do q = 1, size(delta(layer)%onedarray)
                  der_coordinates_weights_atomic_output(p, q)= 0.0d0
              end do
          end do
          do p = 1, size(der_coordinates_ohat)
              do q = 1, size(delta(layer)%onedarray)
              der_coordinates_weights_atomic_output(p, q) = &
              der_coordinates_weights_atomic_output(p, q) + &
              der_coordinates_ohat(p) * delta(layer)%onedarray(q) + &
              ohat(layer)%onedarray(p)* &
              der_coordinates_delta(layer)%onedarray(q)
              end do
          end do 
          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
              unraveled_der_variables(element)%weights(&
              layer)%twodarray(p, q) = &
              unraveled_variables(element)%slope * &
              der_coordinates_weights_atomic_output(p, q)
              end do
          end do
          deallocate(der_coordinates_ohat)
          deallocate(der_coordinates_weights_atomic_output)
      end do

!     deallocating neural network 
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(delta)
          deallocate(delta(p)%onedarray)
      end do
      deallocate(delta)
      do p = 1, size(D)
          deallocate(D(p)%onedarray)
      end do
      deallocate(D)
      do p = 1, size(der_coordinates_o)
          deallocate(der_coordinates_o(p)%onedarray)
      end do
      deallocate(der_coordinates_o)
      do p = 1, size(der_coordinates_delta)
          deallocate(der_coordinates_delta(p)%onedarray)
      end do
      deallocate(der_coordinates_delta)
      do p = 1, size(der_coordinates_D)
          deallocate(der_coordinates_D(p)%onedarray)
      end do
      deallocate(der_coordinates_D)

      k = 0
      l = 0
      do element = 1, no_of_elements
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            no_of_rows = no_nodes_of_elements(k + j) + 1
            no_of_cols = no_nodes_of_elements(k + j + 1)
            do p = 1, no_of_rows
                do q = 1, no_of_cols
                    get_variable_der_of_forces(&
                    l + (p - 1) * no_of_cols + q) = &
                    unraveled_der_variables(&
                    element)%weights(j)%twodarray(p, q)
                end do
            end do
            l = l + no_of_rows * no_of_cols
        end do
      end do
      do element = 1, no_of_elements
        get_variable_der_of_forces(l + 2 *  element - 1) = &
        unraveled_der_variables(element)%intercept
        get_variable_der_of_forces(l + 2 * element) = &
        unraveled_der_variables(element)%slope
      end do

!     deallocating derived-type variables      
      do element = 1, no_of_elements
        deallocate(unraveled_variables(element)%weights)
        deallocate(unraveled_der_variables(element)%weights)
      end do
      
      end function get_variable_der_of_forces

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      end module regression

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
