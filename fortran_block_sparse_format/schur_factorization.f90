! -----------------------------------------------------------------------------
! LU factorization of 2x2 block matrices using Schur complement
! -----------------------------------------------------------------------------
module schur_factorization_module
    implicit none

    contains

    subroutine schurfact(A, ipiv, m, n, factorized, status)
        ! Imports
        use kind_module,            only: mklint, wp => long
        use linear_algebra_module,  only: mul_safe, lu_safe, tri_solve_safe, permute_rows
        use optional_values_module, only: optval
        use error_handler_module

        ! Arguments
        complex(wp),                        intent(inout) :: A(:,:)
            !! Matrix to be factorized
        integer(mklint),                    intent(inout) :: ipiv(:)
            !! Row pivot indices
        integer,                            intent(in) :: m
            !! Size of first block partition
        integer, optional,                  intent(in) :: n
            !! Total number of rows/columns in the matrix
        logical, optional,                  intent(in) :: factorized
            !! Set to true if A_11 is already factorized
        type(error_handler_type), optional, intent(inout) :: status
            !! Error handler

        ! Variables
        integer :: num_unk
        logical :: factorized_

        ! Parameters
        character(len=*), parameter :: method = "schurfact"

        factorized_ = optval(factorized, .false.)
        num_unk = optval(n, min(size(A,1), size(A,2)))

        ! Check sizes
        if (m > num_unk) then
            call throw( &
                type=error_type_internal, &
                method=method, &
                message='Incompatible sizes', &
                handler=status)
            return
        end if

        ! Factor A_11 = L_11 U_11 (if not already factorized)
        if (not(factorized_)) then
            call lu_safe( &
                A=A(1:m, 1:m), &
                ipiv=ipiv(1:m), &
                status=status &
                )
            if (err(status, at=_AT_)) return
        end if

        ! Just a regular LU if one block is requested
        if (m == num_unk) return

        ! Compute L_21 = A_21 U_11^-1 and U_12 = L_11^−1 A_12.
        call tri_solve_safe( &
            B=A(m+1:num_unk, 1:m), &
            A=A(1:m, 1:m), &
            side='R', &
            uplo='U', &
            diag="N", &
            status=status &
            )
        if (err(status, at=_AT_)) return

        call tri_solve_safe( &
            B=A(1:m, m+1:num_unk), &
            A=A(1:m, 1:m), &
            side='L', &
            uplo='L', &
            diag='U', &
            pivot=ipiv(1:m), &
            status=status &
            )
        if (err(status, at=_AT_)) return

        ! Compute the Schur complement S = A_22 − L_21 U_12 and factor L_22 U_22 = S.
        call mul_safe( &
            C=A(m+1:num_unk, m+1:num_unk), &
            A=A(m+1:num_unk, 1:m), &
            B=A(1:m, m+1:num_unk), &
            alpha=cmplx(-1, kind=wp), &
            beta=cmplx(1, kind=wp), &
            status=status &
            )
        if (err(status, at=_AT_)) return

        call lu_safe( &
            A=A(m+1:num_unk, m+1:num_unk), &
            ipiv=ipiv(m+1:num_unk), &
            status=status &
            )
        if (err(status, at=_AT_)) return

        call permute_rows(A(m+1:num_unk, 1:m), ipiv(m+1:num_unk), &
            .false., status)
        if (err(status, at=_AT_)) return

        ipiv(m+1:num_unk) = ipiv(m+1:num_unk) + m

    end subroutine schurfact

end module schur_factorization_module
