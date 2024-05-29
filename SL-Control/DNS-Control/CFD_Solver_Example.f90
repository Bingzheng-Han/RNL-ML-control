program main
    ! This file is not a real CFD solver.
    ! It is only used as an example showing how to call PYTHON control model in CFD.

    use variables
    use, intrinsic :: iso_c_binding
    implicit none
    interface
      subroutine pymodel(state,n1, n2,step,n3,n4,reward,n5,n6) bind(c)
       use iso_c_binding
       integer(c_int) :: n1,n2,n3,n4,n5,n6
       real(c_double) :: state(n1,n2),step,reward(n5,n6)
       end subroutine pymodel
    end interface
    
    integer ntime
    REAL(8) INITIALX(128,128),INITIALY(128,128)
    logical(kind=4) :: isExist

    ! call setup...
    ! some subroutines of calling settings of CFD solver

    Call pymodel(INITIALX,128,128,0.0,1,1,INITIALY,128,128) ! Used for initialing the PYTHON control model

    ! begin the CFD calculating
    do ntime = 1, ntst

        call bcond ! wall blowing and suction is applied on the bottom wall

        ! call solver...
        ! some subroutines of calling CFD solver

    end do

end program main

!***************** BCOND *********************** 
subroutine bcond
    ! this subroutine is used to obtain the blowing and suction on the bottom wall

    use variables
    use, intrinsic :: iso_c_binding
    implicit none
    interface
        subroutine pymodel(state,n1, n2,step,n3,n4,reward,n5,n6) bind(c)
        use iso_c_binding
        integer(c_int) :: n1,n2,n3,n4,n5,n6
        real(c_double) :: state(n1,n2),step,reward(n5,n6)
        end subroutine pymodel
    end interface

    integer i,j,k
    Real(8) C, avg, FIRMS
    Real(8) DUDY(128,128),blow(128,128)

    !------------GET DUDY-------------
    Do I=1,128
        Do K=1,128
            ! DUDY(I,k)=...
        enddo
    enddo

    !------------GET BLOW-------------
    call pymodel(DUDY,128,128,real(ntime),1,1,blow,128,128)

    !------------FC1-------------
    avg=0.0
    FIRMS=0.0
    Do I=1,128
        Do K=1,128
            avg=avg+blow(I,K)
            FIRMS=FIRMS+(blow(I,K))**2
        ENDDO
    enddo
    avg=avg/128/128
    FIRMS=sqrt(FIRMS/128/128)
    C=0.15*0.068/FIRMS ! The constant C is used to adjust the control amplitude

    Do I=1,128
        Do K=1,128
            blow(I,K)=C*(blow(I,K)-avg)
        enddo
    enddo

end subroutine bcond
