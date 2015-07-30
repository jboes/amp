!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
!     Fortran Version = 1
      subroutine check_version(version, warning) 
      implicit none
    
      integer :: version, warning
!f2py         intent(in) :: version
!f2py         intent(out) :: warning
      if (version .NE. 1) then
          warning = 1
      else
          warning = 0
      end if     
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module containing all the data of fingerprints (should be feed in by python)
      module fingerprint_props
      implicit none
      
      double precision, allocatable:: &
      raveled_fingerprints_of_images(:, :)
      integer, allocatable:: len_fingerprints_of_elements(:)
      double precision, allocatable:: raveled_der_fingerprints(:, :)
      double precision, allocatable::min_fingerprints(:, :)
      double precision, allocatable::max_fingerprints(:, :)

      end module fingerprint_props

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module containing all the data of images (should be feed in by python)
      module images_props
      implicit none
      
      double precision:: energy_coefficient
      double precision:: force_coefficient
      logical:: train_forces
      logical:: fingerprinting
      integer:: no_procs
      integer, allocatable:: no_sub_images(:)
      integer:: no_of_images
!     fingerprinting variables
      integer:: no_of_elements
      integer, allocatable:: elements_numbers(:)
      double precision, allocatable:: real_energies(:)
      double precision, allocatable:: real_forces_of_images(:, :)
      integer, allocatable:: no_of_atoms_of_images(:)
      integer, allocatable:: atomic_numbers_of_images(:)
      integer, allocatable:: list_of_no_of_neighbors(:)
      integer, allocatable:: raveled_neighborlists(:)
!     no fingerprinting variables
      integer:: no_of_atoms_of_image
      double precision, allocatable:: atomic_positions_of_images(:, :)
      
      end module images_props

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     subroutine that shares the task corresponding to calculation of cost function and its derivative
!     between cores
      subroutine share_cost_function_task_between_cores(proc, &
      variables, len_of_variables, energy_square_error, &
      force_square_error, der_variables_square_error)

      use images_props
      use fingerprint_props
      use regression

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! input/output variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      integer:: proc
      integer:: len_of_variables
      double precision:: variables(len_of_variables)
      double precision:: energy_square_error, force_square_error
      double precision:: der_variables_square_error(len_of_variables)
!f2py         intent(in):: variables, proc
!f2py         intent(out):: energy_square_error, force_square_error
!f2py         intent(out):: der_variables_square_error

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! type definition !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type:: image_forces
        sequence
        double precision, allocatable:: atomic_forces(:, :)
      end type image_forces

      type:: integer_one_d_array
        sequence
        integer, allocatable:: onedarray(:)
      end type integer_one_d_array

      type:: embedded_real_one_one_d_array
        sequence
        type(real_one_d_array), allocatable:: onedarray(:)
      end type embedded_real_one_one_d_array

      type:: embedded_real_one_two_d_array
        sequence
        type(real_two_d_array), allocatable:: onedarray(:)
      end type embedded_real_one_two_d_array

      type:: embedded_integer_one_one_d_array
        sequence
        type(integer_one_d_array), allocatable:: onedarray(:)
      end type embedded_integer_one_one_d_array

      type:: embedded_one_one_two_d_array
        sequence
        type(embedded_real_one_two_d_array), allocatable:: onedarray(:)
      end type embedded_one_one_two_d_array

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! dummy variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      double precision, allocatable :: fingerprint(:)
      type(embedded_real_one_one_d_array):: &
      unraveled_fingerprints_of_images(no_sub_images(proc))
      type(integer_one_d_array):: &
      unraveled_atomic_numbers_of_images(no_sub_images(proc))
      integer, allocatable:: unraveled_atomic_numbers(:)
      double precision:: amp_energy, real_energy, atomic_amp_energy
      double precision:: force, temp
      integer:: i, index, j, p, k, q, l, m, &
      len_of_fingerprint, symbol, element, image_no, no_of_atoms, &
      len_of_input
      double precision:: &
      partial_der_variables_square_error(len_of_variables)
      type(image_forces):: unraveled_real_forces(no_sub_images(proc))
      type(embedded_integer_one_one_d_array):: &
      unraveled_neighborlists(no_sub_images(proc))
      type(embedded_one_one_two_d_array):: &
      unraveled_der_fingerprints_of_images(no_sub_images(proc))
      double precision, allocatable:: der_fingerprint(:)
      integer:: n_index, n_symbol, self_index
      integer:: first_image_no, no_proc_images
      double precision, allocatable:: &
      real_forces(:, :), amp_forces(:, :)
      integer, allocatable:: n_self_indices(:)
!     no fingerprinting scheme
      type(real_one_d_array):: &
      unraveled_atomic_positions(no_sub_images(proc))
      double precision :: input(3 * no_of_atoms_of_image)
      double precision :: input_(3 * no_of_atoms_of_image)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! calculations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      first_image_no = 1
      do i = 1, proc - 1
        first_image_no = first_image_no + no_sub_images(i)
      end do
      no_proc_images = no_sub_images(proc)
      if (fingerprinting .eqv. .false.) then
        call unravel_atomic_positions(first_image_no, no_proc_images)
      else
        call unravel_atomic_numbers(first_image_no, no_proc_images)
        call unravel_fingerprints(first_image_no, no_proc_images)
        call scale_fingerprints(no_proc_images)
      end if
      if (train_forces .eqv. .true.) then
          call unravel_real_forces(first_image_no, no_proc_images)
          if (fingerprinting .eqv. .true.) then
              call unravel_neighborlists(first_image_no, no_proc_images)
              call &
              unravel_der_fingerprints(first_image_no, no_proc_images)
              call scale_der_fingerprints(no_proc_images)
          end if
      end if

      energy_square_error = 0.0d0
      force_square_error = 0.0d0
      do j = 1, len_of_variables
        der_variables_square_error(j) = 0.0d0
      end do

!     summation over images
      do image_no = 1, no_proc_images
        real_energy = real_energies(first_image_no - 1 + image_no)
        do j = 1, len_of_variables
            partial_der_variables_square_error(j) = 0.0d0
        end do
        if (fingerprinting .eqv. .false.) then
            no_of_atoms = no_of_atoms_of_image
            len_of_input = 3 * no_of_atoms
            input = unraveled_atomic_positions(image_no)%onedarray
            amp_energy = &
            get_energy_(len_of_input, input, len_of_variables, &
            variables)
        else
            no_of_atoms = &
            no_of_atoms_of_images(first_image_no - 1 + image_no)
            allocate(unraveled_atomic_numbers(no_of_atoms))
            do p = 1, no_of_atoms
                unraveled_atomic_numbers(p) = &
                unraveled_atomic_numbers_of_images(&
                image_no)%onedarray(p)
            end do
            amp_energy = 0.0d0
            do index = 1, no_of_atoms
                symbol = unraveled_atomic_numbers(index)
                do element = 1, no_of_elements
                    if (symbol == elements_numbers(element)) then
                        exit
                    end if
                end do
                len_of_fingerprint = &
                len_fingerprints_of_elements(element)
                allocate(fingerprint(len_of_fingerprint))
                do p = 1, len_of_fingerprint
                    fingerprint(p) = &
                    unraveled_fingerprints_of_images(&
                    image_no)%onedarray(index)%onedarray(p)
                end do
                atomic_amp_energy = get_energy(symbol, &
                len_of_fingerprint, fingerprint, &
                no_of_elements, elements_numbers, &
                len_of_variables, variables)
                deallocate(fingerprint)
                amp_energy = amp_energy + atomic_amp_energy
            end do
        end if

        energy_square_error = energy_square_error + &
        (amp_energy - real_energy) ** 2.0d0 / (no_of_atoms ** 2.0d0)
        
        if (fingerprinting .eqv. .false.) then
            partial_der_variables_square_error = &
            get_variable_der_of_energy_(len_of_input,input, &
            len_of_variables, variables)
            do j = 1, len_of_variables
                der_variables_square_error(j) = &
                der_variables_square_error(j) + &
                energy_coefficient *  2. * (amp_energy - real_energy) &
                * partial_der_variables_square_error(j) &
                / (no_of_atoms ** 2.0d0)
            end do
        else
            do index = 1, no_of_atoms
                symbol = unraveled_atomic_numbers(index)
                do element = 1, no_of_elements
                    if (symbol == elements_numbers(element)) then
                        exit
                    end if
                end do
                len_of_fingerprint = &
                len_fingerprints_of_elements(element)
                allocate(fingerprint(len_of_fingerprint))
                do p = 1, len_of_fingerprint
                    fingerprint(p) = &
                    unraveled_fingerprints_of_images(&
                    image_no)%onedarray(index)%onedarray(p)
                end do
                partial_der_variables_square_error = &
                get_variable_der_of_energy(symbol, len_of_fingerprint, &
                fingerprint, no_of_elements, elements_numbers, &
                len_of_variables, variables)
                deallocate(fingerprint)
                do j = 1, len_of_variables
                    der_variables_square_error(j) = &
                    der_variables_square_error(j) + &
                    energy_coefficient *  2. * &
                    (amp_energy - real_energy) * &
                    partial_der_variables_square_error(j) / &
                    (no_of_atoms ** 2.0d0)
                end do
            end do
        end if
        
        if (train_forces .eqv. .true.) then
            allocate(real_forces(no_of_atoms, 3))
            allocate(amp_forces(no_of_atoms, 3))
            do self_index = 1, no_of_atoms
                do i = 1, 3
                    real_forces(self_index, i) = &
                    unraveled_real_forces(&
                    image_no)%atomic_forces(self_index, i)
                    amp_forces(self_index, i) = 0.0d0
                end do
            end do
           
            do self_index = 1, no_of_atoms
                if (fingerprinting .eqv. .false.) then
                    do i = 1, 3
                        do p = 1,  3 * no_of_atoms_of_image
                            input_(p) = 0.0d0
                        end do
                        input_(3 * (self_index - 1) + i) = 1.0d0
                        force = get_force_(len_of_input, input, &
                        input_, len_of_variables, variables)
                        amp_forces(self_index, i) = force
                    end do
                else
                    allocate(n_self_indices(size(&
                    unraveled_neighborlists(image_no)%onedarray(&
                    self_index)%onedarray)))
                    do p = 1, size(unraveled_neighborlists(&
                    image_no)%onedarray(self_index)%onedarray)
                        n_self_indices(p) = unraveled_neighborlists(&
                        image_no)%onedarray(self_index)%onedarray(p)
                    end do 
                    do i = 1, 3
                        do l = 1, size(n_self_indices)
                            n_index = n_self_indices(l)
                            n_symbol = unraveled_atomic_numbers(n_index)
                            do element = 1, no_of_elements
                                if (n_symbol == &
                                elements_numbers(element)) then
                                    exit
                                end if
                            end do 
                            len_of_fingerprint = &
                            len_fingerprints_of_elements(element)
                            allocate(der_fingerprint(&
                            len_of_fingerprint))
                            do p = 1, len_of_fingerprint
                                der_fingerprint(p) = &
                                unraveled_der_fingerprints_of_images(&
                                image_no)%onedarray(&
                                self_index)%onedarray(l)%twodarray(i, p)
                            end do
                            allocate(fingerprint(len_of_fingerprint))
                            do p = 1, len_of_fingerprint
                                fingerprint(p) = &
                                unraveled_fingerprints_of_images(&
                                image_no)%onedarray(&
                                n_index)%onedarray(p)
                            end do
                            force = get_force(n_symbol, &
                            len_of_fingerprint, fingerprint, &
                            der_fingerprint, &
                            no_of_elements, elements_numbers, &
                            len_fingerprints_of_elements, &
                            len_of_variables, variables)
                            amp_forces(self_index, i) = &
                            amp_forces(self_index, i) + force
                            deallocate(fingerprint)
                            deallocate(der_fingerprint)
                        end do
                    end do
                end if 

                do i = 1, 3
                    force_square_error = force_square_error + &
                    (1.0d0 / 3.0d0)*(amp_forces(self_index, i)  - &
                    real_forces(self_index, i)) ** 2.0 / no_of_atoms
                end do

                do i = 1, 3
                    if (fingerprinting .eqv. .false.) then
                        do p = 1,  3 * no_of_atoms_of_image
                            input_(p) = 0.0d0
                        end do
                        input_(3 * (self_index - 1) + i) = 1.0d0
                        partial_der_variables_square_error = &
                        get_variable_der_of_forces_(len_of_input, &
                        input, input_, len_of_variables, variables)
                        do j = 1, len_of_variables
                            der_variables_square_error(j) = &
                            der_variables_square_error(j) + &
                            force_coefficient * (2.0d0 / 3.0d0) * &
                            (- amp_forces(self_index, i) + &
                            real_forces(self_index, i)) * &
                            partial_der_variables_square_error(j) / &
                            no_of_atoms
                        end do
                    else
                        do l = 1, size(n_self_indices)
                            n_index = n_self_indices(l)
                            n_symbol = unraveled_atomic_numbers(n_index)
                            do element = 1, no_of_elements
                                if (n_symbol == &
                                elements_numbers(element)) then
                                    exit
                                end if
                            end do 
                            len_of_fingerprint = &
                            len_fingerprints_of_elements(element)
                            allocate(fingerprint(len_of_fingerprint))
                            do p = 1, len_of_fingerprint
                                fingerprint(p) = &
                                unraveled_fingerprints_of_images(&
                                image_no)%onedarray(&
                                n_index)%onedarray(p)
                            end do
                            allocate(der_fingerprint(&
                            len_of_fingerprint))
                            do p = 1, len_of_fingerprint
                                der_fingerprint(p) = &
                                unraveled_der_fingerprints_of_images(&
                                image_no)%onedarray(&
                                self_index)%onedarray(l)%twodarray(i, p)
                            end do
                            partial_der_variables_square_error = &
                            get_variable_der_of_forces(n_symbol, &
                            len_of_fingerprint, fingerprint, &
                            der_fingerprint, &
                            no_of_elements, &
                            elements_numbers, &
                            len_fingerprints_of_elements, &
                            len_of_variables, variables)
                            deallocate(fingerprint)
                            deallocate(der_fingerprint)
                            do j = 1, len_of_variables
                                der_variables_square_error(j) = &
                                der_variables_square_error(j) + &
                                force_coefficient * (2.0d0 / 3.0d0) * &
                                (- amp_forces(self_index, i) + &
                                real_forces(self_index, i)) * &
                                partial_der_variables_square_error(j) &
                                / no_of_atoms
                            end do
                        end do
                    end if
                end do
                if (fingerprinting .eqv. .true.) then
                    deallocate(n_self_indices)
                end if
            end do
                  
            deallocate(real_forces)
            deallocate(amp_forces)
        end if

        if (fingerprinting .eqv. .true.) then     
            deallocate(unraveled_atomic_numbers)
        end if
      end do

!     deallocations for all images
      if (fingerprinting .eqv. .false.) then
        do image_no = 1, no_proc_images
            deallocate(unraveled_atomic_positions(image_no)%onedarray)
        end do
      else
        do image_no = 1, no_proc_images
            deallocate(unraveled_atomic_numbers_of_images(&
            image_no)%onedarray)
        end do
        do image_no = 1, no_proc_images
            no_of_atoms = &
            no_of_atoms_of_images(first_image_no - 1 + image_no)
            do index = 1, no_of_atoms
                deallocate(unraveled_fingerprints_of_images(&
                image_no)%onedarray(index)%onedarray)
            end do
            deallocate(unraveled_fingerprints_of_images(&
            image_no)%onedarray)
        end do
      end if

      if (train_forces .eqv. .true.) then
        do image_no = 1, no_proc_images
            deallocate(unraveled_real_forces(image_no)%atomic_forces)
        end do
        if (fingerprinting .eqv. .true.) then 
            do image_no = 1, no_proc_images
                no_of_atoms = &
                no_of_atoms_of_images(first_image_no - 1 + image_no)
                do self_index = 1, no_of_atoms
                    do n_index = 1, &
                    size(unraveled_der_fingerprints_of_images(&
                    image_no)%onedarray(self_index)%onedarray)
                        deallocate(&
                        unraveled_der_fingerprints_of_images(&
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
                no_of_atoms = &
                no_of_atoms_of_images(first_image_no - 1 + image_no)
                do index = 1, no_of_atoms
                    deallocate(unraveled_neighborlists(&
                    image_no)%onedarray(index)%onedarray)
                end do
                deallocate(unraveled_neighborlists(image_no)%onedarray)
            end do
        end if
      end if

      contains

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     used only in the no-fingerprinting scheme
      subroutine unravel_atomic_positions(first_image_no, &
      no_proc_images)
 
      integer:: first_image_no, no_proc_images

      do image_no = 1, no_proc_images
        allocate(unraveled_atomic_positions(image_no)%onedarray(&
        3 * no_of_atoms_of_image))
        do index = 1, no_of_atoms_of_image
            do i = 1, 3
                unraveled_atomic_positions(image_no)%onedarray(&
                3 * (index - 1) + i) = atomic_positions_of_images(&
                first_image_no + image_no - 1, 3 * (index - 1) + i)
             end do
        end do
      end do
      
      end subroutine unravel_atomic_positions

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine unravel_atomic_numbers(first_image_no, no_proc_images)
 
      integer:: first_image_no, no_proc_images

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
        allocate(unraveled_atomic_numbers_of_images(&
        image_no)%onedarray(no_of_atoms))
        do l = 1, no_of_atoms
            unraveled_atomic_numbers_of_images(image_no)%onedarray(l) &
            = atomic_numbers_of_images(k + l)
        end do
        k = k + no_of_atoms
      end do
      
      end subroutine unravel_atomic_numbers

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
      subroutine unravel_neighborlists(first_image_no, no_proc_images)

      integer:: first_image_no, no_proc_images

      k = 0
      q = 0
      if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            no_of_atoms = no_of_atoms_of_images(image_no)
            do index = 1, no_of_atoms
                q = q + list_of_no_of_neighbors(k + index)
            end do
            k = k + no_of_atoms
        end do
      end if
      do image_no = 1, no_proc_images
        no_of_atoms = &
        no_of_atoms_of_images(first_image_no - 1 + image_no)
        allocate(unraveled_neighborlists(image_no)%onedarray(&
        no_of_atoms))
        do index = 1, no_of_atoms
            allocate(unraveled_neighborlists(image_no)%onedarray(&
            index)%onedarray(list_of_no_of_neighbors(k + index)))
            do p = 1, list_of_no_of_neighbors(k + index)
                unraveled_neighborlists(image_no)%onedarray(&
                index)%onedarray(p) = raveled_neighborlists(q + p)+1
            end do
            q = q + list_of_no_of_neighbors(k + index)
        end do
        k = k + no_of_atoms
      end do
      
      end subroutine unravel_neighborlists

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_real_forces(first_image_no, no_proc_images)

      integer:: first_image_no, no_proc_images

      k = 0
       if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            if (fingerprinting .eqv. .false.) then
                no_of_atoms = no_of_atoms_of_image
            else
                no_of_atoms = no_of_atoms_of_images(image_no)
            end if
            k = k + no_of_atoms
        end do
      end if
      do image_no = 1, no_proc_images
        if (fingerprinting .eqv. .false.) then
            no_of_atoms = no_of_atoms_of_image
        else
            no_of_atoms = &
            no_of_atoms_of_images(first_image_no - 1 + image_no)
        end if
        allocate(unraveled_real_forces(image_no)%atomic_forces(&
        no_of_atoms, 3))
        do index = 1, no_of_atoms
            do i = 1, 3
                unraveled_real_forces(image_no)%atomic_forces(&
                index, i) = real_forces_of_images(k + index, i)
            end do
        end do
        k = k + no_of_atoms
      end do
      
      end subroutine unravel_real_forces
 
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_fingerprints(first_image_no, no_proc_images)
      
      integer:: first_image_no, no_proc_images

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
            do element = 1, no_of_elements
                if (unraveled_atomic_numbers_of_images(&
                image_no)%onedarray(index)== &
                elements_numbers(element)) then
                    allocate(unraveled_fingerprints_of_images(&
                    image_no)%onedarray(index)%onedarray(&
                    len_fingerprints_of_elements(element))) 
                    exit
                end if
            end do
            do l = 1, len_fingerprints_of_elements(element)
                unraveled_fingerprints_of_images(&
                image_no)%onedarray(index)%onedarray(l) = &
                raveled_fingerprints_of_images(k + index, l)
            end do
        end do
      k = k + no_of_atoms
      end do
      
      end subroutine unravel_fingerprints
      
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine scale_fingerprints(no_proc_images)
      
      integer:: no_proc_images

      do image_no = 1, no_proc_images
        do index = 1, size(unraveled_fingerprints_of_images(&
        image_no)%onedarray)
            do element = 1, no_of_elements
                if (unraveled_atomic_numbers_of_images(&
                image_no)%onedarray(index)== &
                elements_numbers(element)) then
                    exit
                end if
            end do    
            do l = 1, len_fingerprints_of_elements(element)
                if ((max_fingerprints(element, l) - &
                min_fingerprints(element, l)) .GT. &
                (10.0d0 ** (-8.0d0))) then
                    temp = unraveled_fingerprints_of_images(&
                    image_no)%onedarray(index)%onedarray(l)
                    temp = -1.0d0 + 2.0d0 * &
                    (temp - min_fingerprints(element, l)) / &
                    (max_fingerprints(element, l) - &
                    min_fingerprints(element, l))
                    unraveled_fingerprints_of_images(&
                    image_no)%onedarray(index)%onedarray(l) = temp
                endif
            end do
        end do
      end do
      
      end subroutine scale_fingerprints

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_der_fingerprints(first_image_no, &
      no_proc_images)

      integer:: first_image_no, no_proc_images
      integer:: no_of_neighbors

      k = 0
      m = 0
      if (first_image_no .GT. 1) then
        do image_no = 1, first_image_no - 1
            no_of_atoms = no_of_atoms_of_images(image_no)
            do index = 1, no_of_atoms
                m = m + list_of_no_of_neighbors(k + index)
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
            do p = 1, size(unraveled_neighborlists(&
            image_no)%onedarray(self_index)%onedarray)
                n_self_indices(p) = unraveled_neighborlists(&
                image_no)%onedarray(self_index)%onedarray(p)
            end do
            no_of_neighbors = list_of_no_of_neighbors(k + self_index)
            allocate(unraveled_der_fingerprints_of_images(&
            image_no)%onedarray(self_index)%onedarray(no_of_neighbors))
            do n_index = 1, no_of_neighbors
                do n_symbol = 1, no_of_elements
                if (unraveled_atomic_numbers_of_images(&
                image_no)%onedarray(n_self_indices(n_index)) == &
                elements_numbers(n_symbol)) then
                    exit
                end if
                end do
                allocate(unraveled_der_fingerprints_of_images(&
                image_no)%onedarray(self_index)%onedarray(&
                n_index)%twodarray(3, len_fingerprints_of_elements(&
                n_symbol)))
                do p = 1, 3
                    do q = 1, len_fingerprints_of_elements(n_symbol)
                        unraveled_der_fingerprints_of_images(&
                        image_no)%onedarray(self_index)%onedarray(&
                        n_index)%twodarray(p, q) = &
                        raveled_der_fingerprints(&
                        3 * m + 3 * n_index + p - 3, q)
                    end do
                end do
            end do
            deallocate(n_self_indices)
            m = m + no_of_neighbors
        end do
        k = k + no_of_atoms
      end do
      
      end subroutine unravel_der_fingerprints

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine scale_der_fingerprints(no_proc_images)
      
      integer:: no_proc_images

      do image_no = 1, no_proc_images
        do self_index = 1, size(unraveled_der_fingerprints_of_images(&
        image_no)%onedarray)
            allocate(n_self_indices(size(&
            unraveled_neighborlists(image_no)%onedarray(&
            self_index)%onedarray)))
            do p = 1, size(unraveled_neighborlists(image_no)%onedarray(&
            self_index)%onedarray)
                n_self_indices(p) = unraveled_neighborlists(&
                image_no)%onedarray(self_index)%onedarray(p)
            end do
            do n_index = 1, size(n_self_indices)
                do n_symbol = 1, no_of_elements
                if (unraveled_atomic_numbers_of_images(&
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
                            temp = &
                            unraveled_der_fingerprints_of_images(&
                            image_no)%onedarray(self_index)%onedarray(&
                            n_index)%twodarray(p, q)
                            temp = 2.0d0 * temp / &
                            (max_fingerprints(n_symbol, q) - &
                            min_fingerprints(n_symbol, q))
                            unraveled_der_fingerprints_of_images(&
                            image_no)%onedarray(self_index)%onedarray(&
                            n_index)%twodarray(p, q) = temp
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