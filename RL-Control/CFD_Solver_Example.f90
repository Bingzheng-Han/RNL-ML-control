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
    REAL(8) INITIALX(16,128),INITIALY(16,128)
    logical(kind=4) :: isExist

    ! call setup...
    ! some subroutines of calling settings of CFD solver

    ! Obtain the action of the first time step
    ! Used for initialing the PYTHON control model
    do i = 1, 16
        do k = 1, 128
            state(I,K)=DWDY(I,K)
        end do
    end do
    call pymodel(state,16,128,action,16,128,REWARD,1,1,state,16,128,real(NTIME),1,1)
    action = action

    ! begin the CFD calculating
    do ntime = 1, ntst

        call bcond ! wall blowing and suction is applied on the bottom wall

        ! call solver...
        ! some subroutines of calling CFD solver

        CALL UPDATENN ! update the DDPG

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
    Real(8) blow(16,128)

    !------------GET BLOW-------------
    blow = action

    !------------FC1-------------
    avg=0.0
    FIRMS=0.0
    Do I=1,16
        Do K=1,128
            avg=avg+blow(I,K)
            FIRMS=FIRMS+(blow(I,K))**2
        ENDDO
    enddo
    avg=avg/16/128
    FIRMS=sqrt(FIRMS/16/128)
    C=0.15*0.068/FIRMS ! The constant C is used to adjust the control amplitude

    Do I=1,16
        Do K=1,128
            blow(I,K)=C*(blow(I,K)-avg)
        enddo
    enddo

end subroutine bcond

!***************** UPDATENN *********************** 
subroutine UPDATENN
    use variables

    use, intrinsic :: iso_c_binding
    implicit none
    interface
      subroutine pymodel(s,n1,n2,a,n3,n4,r,n5,n6,s_,n7,n8,step,n9,n10) bind(c)
      use iso_c_binding
      integer(c_int) :: n1,n2,n3,n4,n5,n6,n7,n8,n9,n10
      real(c_double) :: s(n1,n2),a(n3,n4),r,s_(n7,n8),step
      end subroutine pymodel
    end interface
    INTEGER I, J, K, IL
    Real(8) CD_epoch, DWDY(16,128)

    do i = 1, 16
        do k = 1, 128
            ! DWDY(I,K)=...
        end do
    end do

    REWARD_period = REWARD_period - (WSMBOTTOM - WSMTOP) / WSMTOP
    if (mod(NTIME,state_step)==0) then ! state_step=50
        REWARD=REWARD_period / state_step
        Reward_epoch(mod(NTIME/state_step,ep_step))=REWARD
        call pymodel(state,16,128,action,16,128,REWARD,1,1,DWDY,16,128,real(NTIME),1,1)
        state = DWDY ! here the state is the next state (s_) after the action 
        action = action
        REWARD_period = 0.0
    endif

    IF (mod(NTIME,EPISODE_LEN)==0)THEN ! EPISODE_LEN=800,1200,2000 for Ret=100,180,950
        ! here the environment will be reset

        CALL READUP ! reset the velocity and pressure fields
        do i = 1, 16
            do k = 1, 128
                ! DWDY(I,K)=...
                state(I,K)=DWDY(I,K)
            end do
        end do

    endif

end subroutine UPDATENN
