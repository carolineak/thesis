module block_sparse_format_module

use kind_module, only: wp => long, long
use error_handler_module, only: error_handler_type, err, error_type_allocation
use range_module, only: int_range, range
use linear_algebra_module, only: mul
use matrix_factorization_module, only: lu, tri_solve

implicit none

type matrix_block
    complex(wp), allocatable :: mat(:,:) ! complex matrix
end type matrix_block

type block_slice
    integer :: num_blocks = 0 ! Number of blocks in slice

    integer, allocatable :: indices(:) ! Array of indeces in slice - used to lookup to blocks

    type(int_range) :: range

end type block_slice

type block_sparse_format
    integer :: m
    integer :: n

    type(matrix_block), allocatable :: blocks(:) ! Array of matrix blocks

    integer :: num_rows ! Number of block rows
    integer :: num_cols ! Number of block columns

    type(block_slice), allocatable :: rows(:) ! Row slices
    type(block_slice), allocatable :: cols(:) ! Column slices

    integer, allocatable :: row_indices(:) ! Row indices of the blocks
    integer, allocatable :: col_indices(:) ! Column indices of the blocks

contains

    procedure, public :: create
    procedure, public :: sparse_matvec
    procedure, public :: sparse_lu

end type block_sparse_format

contains

    subroutine create(this, rows, cols, values, status)

        !! Size of input arrays: 1..num_blocks

        ! Arguments
        class(block_sparse_format), intent(inout) :: this
        integer,                    intent(in)    :: rows(:)
        integer,                    intent(in)    :: cols(:)

        type(matrix_block),         intent(in)    :: values(:)
        type(error_handler_type),   intent(inout) :: status

        ! Internal variables
        integer :: ier
        integer :: i, j
        integer :: num_blocks
        integer :: num_rows, num_cols
        integer :: offset

        num_blocks = size(rows)
        num_rows = maxval(rows)
        num_cols = maxval(cols)

        this%num_rows = num_rows
        this%num_cols = num_cols

        ! Size query
        ! ===================================================================
        ! Count how many blocks each row and column contains
        allocate(this%rows(num_rows), this%cols(num_cols), stat=ier)
        if (err(ier, status, at = _AT_, type = error_type_allocation)) return

        do i = 1, num_blocks
            this%rows(rows(i))%num_blocks = this%rows(rows(i))%num_blocks + 1
            this%cols(cols(i))%num_blocks = this%cols(cols(i))%num_blocks + 1
        end do

        ! Allocate space for block maps
        do i = 1, num_rows
            allocate(this%rows(i)%indices(this%rows(i)%num_blocks), stat=ier)
            if (&
                err(ier, status, at = _AT_, type = error_type_allocation)&
            ) return
        end do

        do i = 1, num_cols
            allocate(this%cols(i)%indices(this%cols(i)%num_blocks), stat=ier)
            if (&
                err(ier, status, at = _AT_, type = error_type_allocation)&
            ) return
        end do

        ! Save original indeces of the blocks
        ! ===================================================================
        allocate(this%row_indices, source=rows, stat=ier)
        if (&
            err(ier, status, at = _AT_, type = error_type_allocation)&
        ) return

        allocate(this%col_indices, source=cols, stat=ier)
        if (&
            err(ier, status, at = _AT_, type = error_type_allocation)&
        ) return

        ! Fill arrays
        ! ===================================================================
        this%rows(:)%num_blocks = 0
        this%cols(:)%num_blocks = 0

        do i = 1, num_blocks
            this%rows(rows(i))%num_blocks = this%rows(rows(i))%num_blocks + 1
            this%rows(rows(i))%indices(this%rows(rows(i))%num_blocks) = i

            this%cols(cols(i))%num_blocks = this%cols(cols(i))%num_blocks + 1
            this%cols(cols(i))%indices(this%cols(cols(i))%num_blocks) = i
        end do

        allocate(this%blocks, source=values, stat=ier)
        if (&
            err(ier, status, at = _AT_, type = error_type_allocation)&
        ) return

        ! Compute size of matrix (m,n)
        ! ===================================================================
        ! Get info of which indices each block row/column corresponds to
        do i = 1, num_blocks
            do j = 1, num_rows
                if (rows(i) /= j) cycle
                this%rows(j)%range = range(1, size(this%blocks(i)%mat, dim=1))
            end do

            do j = 1, num_cols
                if (cols(i) /= j) cycle
                this%cols(j)%range = range(1, size(this%blocks(i)%mat, dim=2))
            end do
        end do

        ! TODO: Make check for corresponding sizes
        offset = 0
        do i = 1, num_rows
            this%rows(i)%range = this%rows(i)%range + offset
            offset = this%rows(i)%range%stop
        end do

        offset = 0
        do i = 1, num_cols
            this%cols(i)%range = this%cols(i)%range + offset
            offset = this%cols(i)%range%stop
        end do

        ! Size of matrix
        this%m = 0
        this%n = 0
        do i = 1, num_rows
            this%m = this%m + this%rows(i)%range%length ! Idea: last index?
        end do

        do i = 1, num_cols
            this%n = this%n + this%rows(i)%range%length
        end do

    end subroutine create


    subroutine sparse_matvec(this, vec_in, vec_out, status)
        !! Compute a sparse matvec

        ! Arguments
        class(block_sparse_format), intent(in) :: this
        complex(wp),            intent(in) :: vec_in(:)
        complex(wp),            intent(inout) :: vec_out(:)
        type(error_handler_type),  intent(inout) :: status

        ! Internal variables
        integer :: i, j
        integer :: block_idx
        type(int_range) :: row_idx
        type(int_range) :: col_idx

        ! Check sizes match
        if (this%n /= size(vec_in)) then
            ! Error
            print *, "Non-compatible sizes"
            return
        end if

        vec_out = 0

        ! Loop over the blocks in the rows and multiply with appropriate range of input vector
        do j = 1, this%num_rows
            do i = 1, this%rows(j)%num_blocks
                block_idx = this%rows(j)%indices(i)
                row_idx = this%rows(j)%range
                col_idx = this%cols(this%col_indices(block_idx))%range

                call mul(vec_out=vec_out(row_idx%start:row_idx%stop), mat=this%blocks(block_idx)%mat, vec_in=vec_in(col_idx%start:col_idx%stop), b=cmplx(1, 0, kind=wp), status=status)
                if (status/=0) return
            end do
        end do



    end subroutine sparse_matvec

    subroutine sparse_lu(this, rel_tol, status)
        !! Compute a sparse LU factorization of the block sparse matrix

        ! Arguments
        class(block_sparse_format), intent(inout) :: this
        real(long),                 intent(in)    :: rel_tol
        type(error_handler_type),   intent(inout) :: status

        ! Internal variables
        integer :: i, j, ii, jj, iii, jjj
        integer, allocatable :: diagonal_blocks(:)
        integer :: blk_idx
        integer :: row_idx, col_idx
        integer :: A_22_idx, U_12_idx, L_21_idx
        logical :: blk_is_present

        ! Enforce that the matrix is square
        ! if (size(this, 1) /= size(this, 2)) then
        !     ! Error
        !     print *, 'Non square matrix'
        !     return
        ! end if

        ! Enforce a symmetric block structure (lazy check)
        if (this%num_rows /= this%num_cols) then
            print *, 'Non symmetric block structure'
            return
        end if

        allocate(diagonal_blocks(this%num_rows))

        ! Outer loop: Eliminate blocks along the diagonal
        do i = 1, this%num_rows
            ! ==========================================================
            ! Not needed for a first implementation
            ! ----------------------------------------------------------
            ! Stage 1: Compress fill-ins (if any are present)
            ! --- Insert code ---

            ! Stage 2: Sparsify fill-ins  (if any are present)
            ! --- Insert code ---
            ! ==========================================================

            ! Stage 3: LU factorize diagonal block (i,i)
            ! Find diagonal blocks
            do ii = 1, this%rows(i)%num_blocks
                do j = 1, this%num_cols
                    do jj = 1, this%cols(j)%num_blocks
                        if (i == j .and. this%cols(j)%indices(jj) == this%rows(i)%indices(ii)) then
                            diagonal_blocks(i) = this%rows(i)%indices(ii)
                            ! Could this be simplified?
                        end if
                    end do
                end do
            end do

            ! Factor A_11 = L_11 U_11
            call lu(this%blocks(diagonal_blocks(i))%mat, status=status)

            ! Stage 4: Create LU factors
            ! Compute L_21 = A_21 U_11^-1
            do jj = 1, this%cols(i)%num_blocks
                blk_idx = this%cols(i)%indices(jj) ! the block indices in the ith column
                if (this%row_indices(blk_idx) <= i) cycle
                if (blk_idx == diagonal_blocks(i)) cycle
                call tri_solve(this%blocks(blk_idx)%mat, &
                               this%blocks(diagonal_blocks(i))%mat, &
                               side='R', &
                               uplo='U', &
                               unit_diagonal='N', &
                               status=status)
                ! Seems like there is a problem in tri_solve
                ! The first time it computes wrong with a very small margin but this propagates
            end do

            ! Compute U_12 = L_11^−1 A_12
            do ii = 1, this%rows(i)%num_blocks
                blk_idx = this%rows(i)%indices(ii)
                if (this%col_indices(blk_idx) <= i) cycle
                if (blk_idx == diagonal_blocks(i)) cycle
                call tri_solve(this%blocks(blk_idx)%mat, &
                               this%blocks(diagonal_blocks(i))%mat, &
                               side='L', &
                               uplo='L', &
                               unit_diagonal='U', &
                               status=status)
            end do

            ! Stage 5: Schur complement update
            ! Compute the Schur complement S = A_22 − L_21 U_12 and factor L_22 U_22 = S
            blk_is_present = .false.
            do ii = 1, this%rows(i)%num_blocks
                blk_idx = this%rows(i)%indices(ii)
                if (this%col_indices(blk_idx) <= i) cycle
                if (blk_idx == diagonal_blocks(i)) cycle
                do jj = 1, this%cols(i)%num_blocks
                    blk_idx = this%cols(i)%indices(jj)
                    if (this%row_indices(blk_idx) <= i) cycle
                    if (blk_idx == diagonal_blocks(i)) cycle
                    ! Check if intersecting block exists
                    L_21_idx = this%cols(i)%indices(jj) ! 2 ! 4
                    U_12_idx = this%rows(i)%indices(ii) ! 5 ! 6
                    row_idx = this%row_indices(L_21_idx) ! 3 ! 3
                    col_idx = this%col_indices(U_12_idx) ! 3 ! 3
                    do jjj = 1, size(this%col_indices)
                        if (this%col_indices(jjj) /= col_idx) cycle
                        do iii = jjj, size(this%row_indices)
                            if (this%row_indices(iii) /= row_idx) cycle
                                A_22_idx = iii ! 7 ! 7
                                blk_is_present = .true.
                                ! Escape the loop (not a problem, but redundant loops after this)
                        end do
                    end do
                    if (.not. blk_is_present) then
                        print *, "Block not present"
                        return
                    end if

                    ! If yes, compute complement
                    call mul(mat_C=this%blocks(A_22_idx)%mat, &
                            mat_A=this%blocks(L_21_idx)%mat, &
                            mat_B=this%blocks(U_12_idx)%mat, &
                            a=cmplx(-1, kind=wp), &
                            b=cmplx(1, kind=wp), &
                            status=status)

                end do
            end do
        end do
    end subroutine sparse_lu

end module block_sparse_format_module
