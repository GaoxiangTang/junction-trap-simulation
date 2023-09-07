! f2py -c junction.f90 -m junction
! python -c "import scipy;from junction import trap;trap.pb=scipy.io.loadmat('junction.mat')['pb'];pos,vec=trap.run([[46e-5,0,7e-5]],170,30e6,scipy.io.loadmat('vdc.mat')['vdc'],2e6,10000);from matplotlib import pyplot as plt;plt.plot(pos[:,0,:-1].T);plt.show()"
module trap
    implicit none
    real(8),parameter :: pi = 3.14159265359, e = 1.602176565d-19, mH = 1.6605402d-27
    real(8),parameter :: Ke = 8.9875517873681d9, kB = 1.38064852d-23, hbar = 1.0545718d-34
    real(8),parameter :: mYb = 171*mH,field_coeff = 1000*e/mYb,coulomb_coeff = e**2*Ke/mYb
    integer,parameter :: nf(3) = (/ 401,21,11 /)
    real(8),parameter :: bf(2,3) = (/ (/-1e-3,1e-3/),(/-5e-5,5e-5/),(/5e-5,1e-4/) /),dbf(3) = 1/(bf(2,:)-bf(1,:))
    
    real(8),allocatable :: pb(:,:,:)
contains

subroutine run(N,rN,vN,vRF,fRF,nDC,vDC,t0,nt,npt,pos,vec,t)    
integer,intent(in) :: N, nDC, nt, npt
real(8),intent(in) :: rN(N,3), vN(N,3), vRF, fRF, vDC(nDC), t0
real(8),allocatable :: voltage(:)

integer :: dpt, i, j, k
real(8) :: dt, d(3), wRF, wX, wY, wZ, Vrms, tic, toc, rnd
real(8) :: r1(N,3), r2(N,3), v1(N,3), v2(N,3), a1(N,3), a2(N,3)
real(8),intent(out) :: pos(N,3,npt), vec(N,3,npt), t

real(8) :: LDIRECT(3), detuning, dist, gamma_laser, p_exc, p_spon, p_stim, rndt,rndp,theta,phi
logical :: exci(N)

allocate(voltage(nDC+2))
voltage(2) = 0
voltage(3:) = vDC

write(*,*) 'calculate...'
call cpu_time(tic)    
call random_seed()

wRF = 2*pi*fRF
dt = (1 / fRF) / 100
dpt = nt / (npt - 1)
!Vrms = sqrt(kB * 300 / mH) / 171

LDIRECT(1)=3**-0.5
LDIRECT(2)=3**-0.5
LDIRECT(3)=3**-0.5

r1 = rN
v1 = vN
do i=1,N
    do j=1,2
        !call random_number(rnd)
        !r1(i,j) = bf(1,j)+(rnd+1)*(bf(2,j)-bf(1,j))/3
        !call random_number(rnd)
        !v1(i,j) = (2*rnd-1)*Vrms
    end do
    !r1(i,3) = (bf(1,3)+bf(2,3))/2
    !v1(i,3) = 0
end do

!write(*,*) 'r = ',r1
!write(*,*) 'v = ',v1
t = t0
do k=1,nt
    if (mod(k-1,dpt) == 0) then
        j = (k-1)/dpt + 1
        !te(1,j) = t
        if (mod(k,2) == 0) then
            pos(:,:,j) = r2
            vec(:,:,j) = v2
        else
            pos(:,:,j) = r1
            vec(:,:,j) = v1
        end if
    end if
    if (mod(k,2) == 0) then
        call LeapFrog(r2,r1,v2,v1,a2,a1)
        !call PEFRL(r2,r1,v2,v1,a2,a1)
    else
        call LeapFrog(r1,r2,v1,v2,a1,a2)
        !call PEFRL(r1,r2,v1,v2,a1,a2)
    end if
end do

call cpu_time(toc)
write(*,*)'time: ',toc-tic

write(*,*) 'r = ',r1
write(*,*) 'v = ',v1

contains

subroutine Acc(r, v, a)
    implicit none
    real(8),intent(in) :: r(N,3),v(N,3)
    real(8),intent(out) :: a(N,3)
    
    !call PseudoAcc(r,a)
    call FieldAcc(r,a)
    
    call CoulombAcc(r,a)
    call DampingAcc(r,v,a)
    !call CoolingAvgAcc(r,v,a)
    !call CoolingAcc(r,v,a)
end subroutine

subroutine PseudoAcc(r,a)
    implicit none
    real(8),intent(in) :: r(N,3)
    real(8),intent(out) :: a(N,3)
    
    a(:,1) = -wX**2 * r(:,1)
    a(:,2) = -wY**2 * r(:,2)
    a(:,3) = -wZ**2 * r(:,3)
end subroutine

subroutine FieldAcc(r,a)
    implicit none
    real(8),intent(in) :: r(N,3)
    real(8),intent(out) :: a(N,3)
    integer :: id(3),idx(8)
    real(8) :: coe(8)
    
    voltage(1) = vRF*cos(wRF*t)
    do i=1,N
        d = (nf-1)*(r(i,:)-bf(1,:))*dbf
        id = floor(d)
        d = d - id
        coe(1) = (1-d(1))*(1-d(2))*(1-d(3))
        coe(2) = (1-d(1))*(1-d(2))*d(3)
        coe(3) = (1-d(1))*d(2)*(1-d(3))
        coe(4) = (1-d(1))*d(2)*d(3)
        coe(5) = d(1)*(1-d(2))*(1-d(3))
        coe(6) = d(1)*(1-d(2))*d(3)
        coe(7) = d(1)*d(2)*(1-d(3))
        coe(8) = d(1)*d(2)*d(3)
        coe = field_coeff * coe
        idx(1) = id(3)+id(2)*nf(3)+id(1)*nf(2)*nf(3)+1
        idx(2) = idx(1) + 1
        idx(3) = id(3)+(1+id(2))*nf(3)+id(1)*nf(2)*nf(3)+1
        idx(4) = idx(3) + 1
        idx(5) = id(3)+id(2)*nf(3)+(1+id(1))*nf(2)*nf(3)+1
        idx(6) = idx(5) + 1
        idx(7) = id(3)+(1+id(2))*nf(3)+(1+id(1))*nf(2)*nf(3)+1
        idx(8) = idx(7) + 1
        do j=1,3
            a(i,j) = dot_product(coe,matmul(pb(j+1,idx,:),voltage))
        end do
    end do
end subroutine

subroutine CoulombAcc(r,a)
    implicit none
    real(8),intent(in) :: r(N,3)
    real(8),intent(out) :: a(N,3)

    do i=1,N
        do j=i+1,N
            d = r(i,:) - r(j,:)
            d = (coulomb_coeff / (d(1)*d(1)+d(2)*d(2)+d(3)*d(3))**1.5) * d
            a(i,:) = a(i,:) + d
            a(j,:) = a(j,:) - d
        end do
    end do
end subroutine

subroutine DampingAcc(r,v,a)
    implicit none
    real*8,intent(in) :: r(N,3),v(N,3)
    real*8,intent(out) :: a(N,3)
    
    do i=1,N
        a(i,:) = a(i,:) + 103663 * (1 - 0.211342 * v(i,:))
    enddo
end subroutine

subroutine CoolingAvgAcc(r,v,a)
    implicit none
    real(8),parameter :: lam = 369.5e-9, kL = 2 * pi / lam, recoil = hbar * kL / mYb
    real(8),parameter :: LSATURATION = 0.8, gamma = 1/(8.7e-9), LDETUNING = -0.5*gamma, waist=50e-6
    real(8),intent(in) :: r(N,3),v(N,3)
    real(8),intent(out) :: a(N,3)

    do i=1,N
        dist = dot_product(r(i,:),r(i,:)) - dot_product(LDIRECT,r(i,:))**2
        gamma_laser = LSATURATION * exp(-dist/waist**2)
        detuning = LDETUNING - kL*(v(i,1)*LDIRECT(1)+v(i,2)*LDIRECT(2)+v(i,3)*LDIRECT(3))
        gamma_laser = gamma_laser / (1 + gamma_laser + (2 * detuning / gamma)**2) * gamma / 2
        a(i,:) = a(i,:) + gamma_laser * recoil * LDIRECT
    end do
end subroutine

subroutine CoolingAcc(r,v,a)
    implicit none
    real(8),parameter :: lam = 369.5e-9, kL = 2 * pi / lam, recoil = hbar * kL / mYb
    real(8),parameter :: LSATURATION = 0.8, gamma = 1/(8.7e-9), LDETUNING = -0.5*gamma, waist=50e-6
    real(8),intent(in) :: r(N,3),v(N,3)
    real(8),intent(out) :: a(N,3)

    do i=1,N
        dist = dot_product(r(i,:),r(i,:)) - dot_product(LDIRECT,r(i,:))**2
        gamma_laser = LSATURATION * exp(-dist/waist**2)
        detuning = LDETUNING - kL*(v(i,1)*LDIRECT(1)+v(i,2)*LDIRECT(2)+v(i,3)*LDIRECT(3))
        gamma_laser = gamma_laser / (1 + gamma_laser + (2 * detuning / gamma)**2) * gamma / 2
        if (exci(i) == 0) then
            p_exc = gamma_laser*dt
            call random_number(rnd)
            if (rnd < p_exc)then
                exci(i) = 1
                a(i,:) = a(i,:) + recoil/dt * LDIRECT
            endif
        else
            p_spon = gamma * dt
            p_stim = gamma_laser * dt
            call random_number(rnd)
            if (rnd < p_stim) then
                exci(i) = 0
                a(i,:) = a(i,:) - recoil/dt * LDIRECT
            elseif (rnd < p_stim + p_spon) then
                !atom gets de-excited, spontaneous, in a random direction
                exci(i) = 0
                call random_number(rndt)
                call random_number(rndp)
                theta = 2*pi*rndt
                phi = acos(2*rndp-1) !pi*rndp
                a(i,1) = a(i,1) + recoil/dt * sin(theta) * cos(phi)
                a(i,2) = a(i,2) + recoil/dt * sin(theta) * sin(phi)
                a(i,3) = a(i,3) + recoil/dt * cos(theta)
            endif
        endif
    enddo
end subroutine
    
subroutine LeapFrog(r0, r1, v0, v1, a0, a1)
    implicit none
    real(8),intent(in) :: r0(N,3),v0(N,3),a0(N,3)
    real(8),intent(out) :: r1(N,3),v1(N,3),a1(N,3)

    r1 = r0 + (v0 + 0.5*a0*dt)*dt
    call Acc(r1,v0,a1)
    v1 = v0 + 0.5*(a0+a1)*dt
    t = t + dt
end subroutine

subroutine PEFRL(r0, r1, v0, v1, a0, a1)
    implicit none
    real(8),parameter :: xi = 0.1786178958448091, lambda = -0.2123418310626054, chi = -0.06626458266981849
    real(8),parameter :: coe1 = 0.5-lambda, coe2 = 1-2*(chi+xi)
    real(8),intent(in) :: r0(N,3),v0(N,3),a0(N,3)
    real(8),intent(out) :: r1(N,3),v1(N,3),a1(N,3)
    real(8) :: dt1
    
    dt1 = xi*dt
    r1 = r0 + v0*dt1
    t = t + dt1
    call Acc(r1,v0,a1)
    v1 = v0 + a1*coe1*dt
    
    dt1 = chi*dt
    r1 = r1 + v1*dt1
    t = t + dt1
    call Acc(r1,v1,a1)
    v1 = v1 + a1*lambda*dt
    
    dt1 = coe2*dt
    r1 = r1 + v1*dt1
    t = t + dt1
    call Acc(r1,v1,a1)
    v1 = v1 + a1*lambda*dt
    
    dt1 = chi*dt
    r1 = r1 + v1*dt1
    t = t + dt1
    call Acc(r1,v1,a1)
    v1 = v1 + a1*coe1*dt
    
    dt1 = xi*dt
    r1 = r1 + v1*dt1
    t = t + dt1
end subroutine
    
end subroutine

end module
