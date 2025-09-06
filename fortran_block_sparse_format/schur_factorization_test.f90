module schur_factorization_test_module

    implicit none
    private
    public :: schur_factorization_test_type

    contains
        procedure :: setup
        procedure :: run_case_01 => run_case_factorized
        procedure :: run_case_02 => run_case_pivot
        procedure :: run_case_03 => run_case_diagonal_blocks
        procedure :: run_case_04 => run_case_fixed
        procedure :: run_case_05 => run_case_10x10
    end type schur_factorization_test_type

contains

    subroutine setup(this, status)

        class(schur_factorization_test_type), intent(inout) :: this
        type(error_handler_type), intent(inout) :: status

        call this%setup_cases(status, 1, 5, &
            description = "Test case for schur factorization", &
            alias = "schur_test", &
            tags = "mhg,schur")

    end subroutine setup

    subroutine run_case_factorized(this, status)
        use schur_factorization_module, only: schurfact
        use matrix_factorization_module, only: lu
        use lapack95_mklint, only: getrs

        class(schur_factorization_test_type), intent(inout) :: this
        type(error_handler_type), intent(inout) :: status

        integer :: m, n, i, j
        complex(wp), allocatable :: A(:,:), B(:,:)
        integer(mklint), allocatable :: ipiv(:)
        real(wp), parameter :: tol = 1.0e-5
        logical :: correct

        ! Generate test matrix
        n = 4
        m = 2
        allocate(A(n, n), B(n,n), ipiv(n))
        do i = 1, n
            do j = 1, n
                if (i > j) then
                    A(i,j) = 1
                else if (i == j) then
                    A(i,j) = 10
                else
                    A(i,j) = 0
                end if
            end do
        end do

        ! call schurfact
        B = A
        call schurfact(A, ipiv, m, status=status)

        ! Check correctness
        call getrs(A, ipiv, B)

        correct = .true.
        do i = 1, n
            do j = 1, n
                if (i == j) then
                    if (abs(B(i,j)-1) > tol) correct = .false.
                else
                    if (abs(B(i,j)) > tol) correct = .false.
                end if
            end do
        end do

        call this%assert(correct, "Schurfact does not compute valid factorization")

        ! Fill A again and factorize A_11
        do i = 1, n
            do j = 1, n
                if (i > j) then
                    A(i,j) = 1
                else if (i == j) then
                    A(i,j) = 10
                else
                    A(i,j) = 0
                end if
            end do
        end do
        B = A

        call lu(A(1:m, 1:m), status=status)

        call schurfact(A, ipiv, m, factorized=.true., status=status)

        ! Check correctness
        call getrs(A, ipiv, B)

        correct = .true.
        do i = 1, n
            do j = 1, n
                if (i == j) then
                    if (abs(B(i,j)-1) > tol) correct = .false.
                else
                    if (abs(B(i,j)) > tol) correct = .false.
                end if
            end do
        end do

        call this%assert(correct, "Schurfact does not compute valid factorization")


    end subroutine run_case_factorized

    subroutine run_case_pivot(this, status)
        use schur_factorization_module, only: schurfact
        use lapack95_mklint, only: getrs

        implicit none

        class(schur_factorization_test_type), intent(inout) :: this
        type(error_handler_type), intent(inout) :: status

        integer :: m, n, i, j
        complex(wp), allocatable :: A(:,:), B(:,:)
        integer(mklint), allocatable :: ipiv(:)
        logical :: correct
        real(wp), parameter :: tol = 1.0e-5_wp

        ! Matrix size and block size
        n = 2
        m = 2
        allocate(A(n,n), B(n,n), ipiv(n))

        ! Define the matrix
        A = reshape([ (0.0_wp,0.0), (1.0_wp,0.0), &
                    (1.0_wp,0.0), (0.0_wp,0.0) ], [2,2])

        ! Perform LU factorization
        B = A
        call schurfact(A, ipiv, m, status=status)

        ! Check correctness
        call getrs(A, ipiv, B)

        correct = .true.
        do i = 1, n
            do j = 1, n
                if (i == j) then
                    if (abs(B(i,j)-1) > tol) correct = .false.
                else
                    if (abs(B(i,j)) > tol) correct = .false.
                end if
            end do
        end do

        call this%assert(correct, "Schurfact does not compute valid factorization")

    end subroutine run_case_pivot


    subroutine run_case_diagonal_blocks(this, status)
        use schur_factorization_module, only: schurfact
        use lapack95_mklint, only: getrs

        class(schur_factorization_test_type), intent(inout) :: this
        type(error_handler_type), intent(inout) :: status

        integer :: m, n, i, j
        complex(wp), allocatable :: A(:,:), B(:,:)
        integer(mklint), allocatable :: ipiv(:)
        real(wp), parameter :: tol = 1.0e-5_wp
        logical :: correct

        ! Matrix size and block size
        n = 4
        m = 2
        allocate(A(n, n), B(n, n), ipiv(n))

        A = 0.0_wp
        A(1,1) = 2.0_wp
        A(2,2) = 3.0_wp
        A(3,3) = 5.0_wp
        A(4,4) = 6.0_wp
        A(1,3) = 1.0_wp
        A(2,4) = 1.0_wp
        A(3,1) = 1.0_wp
        A(4,2) = 1.0_wp

        B = A
        call schurfact(A, ipiv, m, status=status)

        ! Check correctness
        call getrs(A, ipiv, B)

        correct = .true.
        do i = 1, n
            do j = 1, n
                if (i == j) then
                    if (abs(B(i,j)-1) > tol) correct = .false.
                else
                    if (abs(B(i,j)) > tol) correct = .false.
                end if
            end do
        end do

        call this%assert(correct, "Schurfact does not compute valid factorization")


    end subroutine run_case_diagonal_blocks

    subroutine run_case_fixed(this, status)
        use schur_factorization_module, only: schurfact
        use matrix_factorization_module, only: tri_solve
        use linear_algebra_module, only: mul
        use lapack95_mklint, only: getrs
        use vectorized_random_module, only: vsl_stream_state, set_stream, randn

        implicit none

        class(schur_factorization_test_type), intent(inout) :: this
        type(error_handler_type), intent(inout) :: status

        integer :: m, n, i, j, ier
        complex(wp), allocatable :: A(:,:), B(:,:)
        integer(mklint), allocatable :: ipiv(:)
        logical :: correct
        real(wp), parameter :: tol = 1.0e-5_wp
        type(vsl_stream_state) :: stream

        ! Matrix size and block size
        n = 50
        m = 25
        allocate(A(n,n), B(n,n), ipiv(n))

        call set_stream(stream, ier=ier)
        if (err(ier, status, at=_AT_)) return

        call randn(A, stream, ier=ier)
        if (err(ier, status, at=_AT_)) return

        B = A

        ! Factorize
        call schurfact(A, ipiv, m, status=status)

        ! Check correctness
        call getrs(A, ipiv, B)

        do i = 1, n
            B(i,i) = B(i,i) - 1.0d0
        end do
        correct = norm2(abs(B)) < tol * n

        call this%assert(correct, "Schurfact does not compute valid factorization")

    end subroutine run_case_fixed

    subroutine run_case_10x10(this, status)
        use schur_factorization_module, only: schurfact
        use lapack95_mklint, only: getrs

        class(schur_factorization_test_type), intent(inout) :: this
        type(error_handler_type), intent(inout) :: status

        integer :: m, n, i, j
        complex(wp), allocatable :: A(:,:), B(:,:)
        integer(mklint), allocatable :: ipiv(:)
        real(wp), parameter :: tol = 1.0e-5_wp
        logical :: correct

        ! Matrix and block size
        n = 10
        m = 4
        allocate(A(n, n), B(n, n), ipiv(n))

        A = 0.0_wp

        do i = 1, m
            A(i, i) = (2.0_wp, 0.0_wp)
        end do
        do i = m+1, n
            A(i, i) = (3.0_wp, 0.0_wp)
        end do

        A(1, 5) = (1.0_wp, 0.0_wp)
        A(2, 6) = (1.0_wp, 0.0_wp)
        A(5, 1) = (1.0_wp, 0.0_wp)
        A(6, 2) = (1.0_wp, 0.0_wp)

        B = A

        call schurfact(A, ipiv, m, status=status)

        ! Check correctness
        call getrs(A, ipiv, B)

        correct = .true.
        do i = 1, n
            do j = 1, n
                if (i == j) then
                    if (abs(B(i,j)-1) > tol) correct = .false.
                else
                    if (abs(B(i,j)) > tol) correct = .false.
                end if
            end do
        end do

        call this%assert(correct, "Schurfact does not compute valid factorization")
    end subroutine run_case_10x10


    !---------------------------------------------------------------!
    ! Helper routines
    !---------------------------------------------------------------!
    subroutine unpack_lu(A_fact, L, U)
        complex(wp), intent(in) :: A_fact(:,:)
        complex(wp), intent(out) :: L(:,:), U(:,:)
        integer :: i, j, n

        n = size(A_fact,1)
        L = 0.0_wp
        U = 0.0_wp

        do i = 1, n
            do j = 1, n
                if (i > j) then
                    L(i,j) = A_fact(i,j)
                else if (i == j) then
                    L(i,j) = 1.0_wp
                    U(i,j) = A_fact(i,j)
                else
                    U(i,j) = A_fact(i,j)
                end if
            end do
        end do
    end subroutine unpack_lu

end module schur_factorization_test_module
