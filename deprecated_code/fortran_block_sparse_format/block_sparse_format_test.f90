module block_sparse_format_test_module

    use vectorized_random_module, only: randn, set_stream, vsl_stream_state
    use kind_module, only: long
    use block_sparse_format_module, only: block_sparse_format, matrix_block, create, wp
    use linear_algebra_module, only: mul
    use matrix_factorization_module, only: lu

    class(block_sparse_format_test_type), intent(inout) :: this
    type(error_handler_type), intent(inout) :: status

    integer :: n, b, i , j
    complex(wp), allocatable :: dense_matrix(:,:)
    type(vsl_stream_state) :: stream
    integer :: ier
    integer, allocatable :: rows(:), cols(:)
    type(matrix_block), allocatable :: values(:)
    type(block_sparse_format) :: sparse_matrix
    complex(wp), allocatable :: vec_in(:), vec_out_sparse(:), vec_out_dense(:), diff(:)
    ! type(block_sparse_format) :: lu
    real(long) :: rel_tol
    real(wp) :: diff_tol
    complex(wp), allocatable :: diff_matrix(:,:)
    complex(wp) :: norm

    print *, "Running test"

    n = 100 ! Matrix size
    b = 25 ! Block size

    ! Generate matrix with the block structure
    ! 1 - 5 -
    ! - 3 6 -
    ! 2 4 7 -
    ! - - - 8

    ! Create dense matrix randomly filled
    allocate(dense_matrix(n,n))
    call set_stream(stream, ier=ier)
    call randn(dense_matrix, stream, ier=ier)

    ! Fill zero-blocks
    do i = 1, b
        do j = 1, b
            dense_matrix(i+b, j) = 0
            dense_matrix(i+3*b, j) = 0
            dense_matrix(i, j+b) = 0
            dense_matrix(i+3*b, j+b) = 0
            dense_matrix(i+3*b, j+2*b) = 0
            dense_matrix(i, j+3*b) = 0
            dense_matrix(i+b, j+3*b) = 0
            dense_matrix(i+2*b, j+3*b) = 0
        end do
    end do

    ! Generate matrix on the block sparse format
    ! Get blocks from dense matrix
    allocate(values(8), rows(8), cols(8))
    values(1)%mat = dense_matrix(1:b, 1:b)
    values(2)%mat = dense_matrix(1+2*b:3*b, 1:b)
    values(3)%mat = dense_matrix(1+b:2*b, 1+b:2*b)
    values(4)%mat = dense_matrix(1+2*b:3*b, 1+b:2*b)
    values(5)%mat = dense_matrix(1:b, 1+2*b:3*b)
    values(6)%mat = dense_matrix(1+b:2*b, 1+2*b:3*b)
    values(7)%mat = dense_matrix(1+2*b:3*b, 1+2*b:3*b)
    values(8)%mat = dense_matrix(1+3*b:4*b, 1+3*b:4*b)

    ! Set block pattern
    rows(1) = 1
    rows(2) = 3
    rows(3) = 2
    rows(4) = 3
    rows(5) = 1
    rows(6) = 2
    rows(7) = 3
    rows(8) = 4

    cols(1) = 1
    cols(2) = 1
    cols(3) = 2
    cols(4) = 2
    cols(5) = 3
    cols(6) = 3
    cols(7) = 3
    cols(8) = 4

    ! Create sparse matrix
    call sparse_matrix%create(rows, cols, values, status=status)

    ! Create vector for testing
    allocate(vec_in(n), vec_out_sparse(n), vec_out_dense(n))
    vec_in = 1

    ! Compute sparse matvec
    call sparse_matrix%sparse_matvec(vec_in, vec_out_sparse, status=status)

    ! Compute dense matvec
    call mul(vec_out_dense, dense_matrix, vec_in, status=status)

    ! Check that the two matrices are the same
    allocate(diff(n))
    diff = vec_out_sparse - vec_out_dense

    ! Compute sparse LU
    rel_tol = 0.1
    call sparse_matrix%sparse_lu(rel_tol, status=status)

    ! Compute dense LU
    call lu(dense_matrix, status=status)

    allocate(diff_matrix(n,n))
    diff_matrix(1:b, 1:b) = dense_matrix(1:b, 1:b) - sparse_matrix%blocks(1)%mat
    diff_matrix(1+2*b:3*b, 1:b) = dense_matrix(1+2*b:3*b, 1:b) - sparse_matrix%blocks(2)%mat
    diff_matrix(1+b:2*b, 1+b:2*b) = dense_matrix(1+b:2*b, 1+b:2*b) - sparse_matrix%blocks(3)%mat
    diff_matrix(1+2*b:3*b, 1+b:2*b) = dense_matrix(1+2*b:3*b, 1+b:2*b) - sparse_matrix%blocks(4)%mat
    diff_matrix(1:b, 1+2*b:3*b) = dense_matrix(1:b, 1+2*b:3*b) - sparse_matrix%blocks(5)%mat
    diff_matrix(1+b:2*b, 1+2*b:3*b) = dense_matrix(1+b:2*b, 1+2*b:3*b) - sparse_matrix%blocks(6)%mat
    diff_matrix(1+2*b:3*b, 1+2*b:3*b) = &
        dense_matrix(1+2*b:3*b, 1+2*b:3*b) - sparse_matrix%blocks(7)%mat
    diff_matrix(1+3*b:4*b, 1+3*b:4*b) = &
        dense_matrix(1+3*b:4*b, 1+3*b:4*b) - sparse_matrix%blocks(8)%mat

    norm = norm2(abs(diff_matrix))/norm2(abs(dense_matrix))
    print *, "norm:", norm

end module block_sparse_format_test_module
